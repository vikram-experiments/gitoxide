//! Change discovery and structural matching for tree merges.
//!
//! This module turns each base-to-side tree diff into a [`SideState`]: a flat
//! [`ChangeList`] containing scheduling state, paired with a [`TreeNodes`] path
//! index whose entries point back into that list. It also finds path and rename
//! interactions between sides and classifies the pairs consumed by the resolver.
//!
//! Semantic conflict resolution and edits to the result tree belong to the
//! sibling `resolve` module; this module only prepares, indexes, and matches work.

use bstr::{BString, ByteSlice, ByteVec};
use gix_diff::tree_with_rewrites::Change;
use gix_error::ResultExt;
use gix_error::{Result, message};
use gix_object::{FindExt, TreeRefIter, tree::EntryRef};
use std::collections::VecDeque;

use crate::tree::utils::{ChangeList, ChangeListRef, PossibleConflict, TreeNodes, track};

pub(super) struct SideState {
    changes: ChangeList,
    tree: TreeNodes,
}

/// Lifecycle
impl SideState {
    fn from_changes(changes: ChangeList) -> Self {
        let mut tree = TreeNodes::new();
        for (idx, change) in changes.iter().enumerate() {
            tree.track_change(&change.inner, idx);
        }
        SideState { changes, tree }
    }
}

impl SideState {
    /// Borrow the change schedule and its path index separately for resolution.
    ///
    /// The resolver swaps these pairs between "ours" and "theirs", appends deferred
    /// changes to the list, and updates the corresponding index while processing them.
    /// Every index stored in `TreeNodes` continues to refer into the returned `ChangeList`.
    pub(super) fn parts_mut(&mut self) -> (&mut ChangeList, &mut TreeNodes) {
        (&mut self.changes, &mut self.tree)
    }
    /// Return whether unrelated rewrite destinations can claim the same source identity.
    ///
    /// Rewrites derived from a parent directory rename have a relation and don't represent
    /// ambiguous identity pairing themselves.
    pub(super) fn has_ambiguous_rewrite_sources(&self) -> bool {
        let mut sources = std::collections::HashSet::new();
        self.changes.iter().any(|change| match &change.inner {
            Change::Rewrite {
                source_id,
                relation: None,
                ..
            } => !sources.insert(*source_id),
            _ => false,
        })
    }

    /// Compare both sides by their ordered `(source, destination)` paths.
    ///
    /// When unrelated rewrites share an object ID, whichever side is scheduled first can
    /// otherwise decide how those identities are paired. The resolver uses this ordering
    /// to choose the same first side after the caller reverses ours and theirs. Paths are
    /// used instead of object IDs so the choice remains stable across hash kinds.
    pub(super) fn cmp_for_scheduling(&self, other: &Self) -> std::cmp::Ordering {
        self.changes
            .iter()
            .map(|change| (change.inner.source_location(), change.inner.location()))
            .cmp(
                other
                    .changes
                    .iter()
                    .map(|change| (change.inner.source_location(), change.inner.location())),
            )
    }
}

#[derive(Debug)]
pub(super) enum MatchKind {
    /// A tree is supposed to be superseded by something else.
    EraseTree,
    /// A leaf node is superseded by a tree.
    EraseLeaf,
}

/// Collect both sides relative to the same base in one breadth-first tree walk.
///
/// Tree entries from the base are decoded once per visited directory and compared with
/// both sides before descending. Each side's resulting changes are then passed through
/// the same rewrite tracker and normalization as an independent tree diff.
#[expect(clippy::too_many_arguments)]
pub(super) fn collect_both(
    base_tree: &gix_hash::oid,
    our_tree: &gix_hash::oid,
    their_tree: &gix_hash::oid,
    base_buf: &mut Vec<u8>,
    diff_state: &mut gix_diff::tree::State,
    objects: &impl gix_object::FindObjectOrHeader,
    diff_resource_cache: &mut gix_diff::blob::Platform,
    rewrites: Option<gix_diff::Rewrites>,
) -> Result<(SideState, SideState)> {
    let (mut our_raw, mut their_raw) = (Vec::new(), Vec::new());
    if base_tree != our_tree || base_tree != their_tree {
        let mut queue = VecDeque::from([Walk {
            base: Some(base_tree.to_owned()),
            ours: Some(our_tree.to_owned()),
            theirs: Some(their_tree.to_owned()),
            ours_relation: None,
            theirs_relation: None,
            path: Default::default(),
            is_root: true,
        }]);
        let (mut our_change_id, mut their_change_id) = (0, 0);
        while let Some(walk) = queue.pop_front() {
            if walk.base == walk.ours && walk.base == walk.theirs {
                continue;
            }
            let base_entries = match walk.base.as_ref() {
                Some(id) if walk.is_root => TreeRefIter::from_bytes(base_buf, id.kind())
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .or_raise(|| message("Tree merge failed"))?,
                Some(id) => read_entries(id, base_buf, objects)?,
                None => Vec::new(),
            };
            let our_entries = match walk.ours.as_ref() {
                Some(id) if walk.base.as_ref() == Some(id) => base_entries.clone(),
                Some(id) => read_entries(id, &mut diff_state.buf1, objects)?,
                None => Vec::new(),
            };
            let their_entries = match walk.theirs.as_ref() {
                Some(id) if walk.base.as_ref() == Some(id) => base_entries.clone(),
                Some(id) => read_entries(id, &mut diff_state.buf2, objects)?,
                None => Vec::new(),
            };
            let (mut base_idx, mut our_idx, mut their_idx) = (0, 0, 0);
            while base_idx < base_entries.len() || our_idx < our_entries.len() || their_idx < their_entries.len() {
                let next = [
                    base_entries.get(base_idx),
                    our_entries.get(our_idx),
                    their_entries.get(their_idx),
                ]
                .into_iter()
                .flatten()
                .min_by(|a, b| a.cmp(b))
                .copied()
                .expect("at least one tree entry remains");
                let base = take_equal(&base_entries, &mut base_idx, next);
                let ours = take_equal(&our_entries, &mut our_idx, next);
                let theirs = take_equal(&their_entries, &mut their_idx, next);

                let mut location = walk.path.clone();
                if !location.is_empty() {
                    location.push(b'/');
                }
                location.push_str(next.filename);
                let ours_child_relation = record_pair(
                    base,
                    ours,
                    walk.ours_relation,
                    &location,
                    &mut our_change_id,
                    &mut our_raw,
                );
                let theirs_child_relation = record_pair(
                    base,
                    theirs,
                    walk.theirs_relation,
                    &location,
                    &mut their_change_id,
                    &mut their_raw,
                );
                let base_child = base
                    .filter(|entry| entry.mode.is_tree())
                    .map(|entry| entry.oid.to_owned());
                let ours_child = ours
                    .filter(|entry| entry.mode.is_tree())
                    .map(|entry| entry.oid.to_owned());
                let theirs_child = theirs
                    .filter(|entry| entry.mode.is_tree())
                    .map(|entry| entry.oid.to_owned());
                if base_child.is_some() || ours_child.is_some() || theirs_child.is_some() {
                    queue.push_back(Walk {
                        base: base_child,
                        ours: ours_child,
                        theirs: theirs_child,
                        ours_relation: ours_child_relation,
                        theirs_relation: theirs_child_relation,
                        path: location,
                        is_root: false,
                    });
                }
            }
        }
    }

    let source_tree = if rewrites.is_some_and(|r| {
        r.copies
            .is_some_and(|c| c.source == gix_diff::rewrites::CopySource::FromSetOfModifiedFilesAndAllSources)
    }) {
        Some(source_tree_changes(base_tree, objects)?)
    } else {
        None
    };
    let ours = finish_changes(our_raw, rewrites, source_tree.as_deref(), diff_resource_cache, objects)?;
    let theirs = finish_changes(
        their_raw,
        rewrites,
        source_tree.as_deref(),
        diff_resource_cache,
        objects,
    )?;
    Ok((SideState::from_changes(ours), SideState::from_changes(theirs)))
}

struct Walk {
    base: Option<gix_hash::ObjectId>,
    ours: Option<gix_hash::ObjectId>,
    theirs: Option<gix_hash::ObjectId>,
    ours_relation: Option<gix_diff::tree::visit::Relation>,
    theirs_relation: Option<gix_diff::tree::visit::Relation>,
    path: BString,
    is_root: bool,
}

fn read_entries<'a>(
    tree_id: &gix_hash::oid,
    buf: &'a mut Vec<u8>,
    objects: &impl gix_object::FindObjectOrHeader,
) -> Result<Vec<EntryRef<'a>>> {
    let mut entries = Vec::new();
    for entry in objects
        .find_tree_iter(tree_id, buf)
        .or_raise(|| message("Tree merge failed"))?
    {
        entries.push(entry.or_raise(|| message("Tree merge failed"))?);
    }
    Ok(entries)
}

fn take_equal<'a>(entries: &'a [EntryRef<'a>], index: &mut usize, next: EntryRef<'a>) -> Option<EntryRef<'a>> {
    entries.get(*index).copied().filter(|entry| {
        if entry.cmp(&next).is_eq() {
            *index += 1;
            true
        } else {
            false
        }
    })
}

fn record_pair(
    base: Option<EntryRef<'_>>,
    side: Option<EntryRef<'_>>,
    relation: Option<gix_diff::tree::visit::Relation>,
    location: &bstr::BString,
    change_id: &mut u32,
    changes: &mut Vec<(gix_diff::tree::visit::Change, BString)>,
) -> Option<gix_diff::tree::visit::Relation> {
    use gix_diff::tree::visit::{Change, Relation};
    let mut child_relation = relation;
    let change = match (base, side) {
        (Some(base), Some(side)) if base.mode.is_tree() && side.mode.is_tree() => {
            (base.oid != side.oid).then(|| Change::Modification {
                previous_entry_mode: base.mode,
                previous_oid: base.oid.to_owned(),
                entry_mode: side.mode,
                oid: side.oid.to_owned(),
            })
        }
        (Some(base), Some(side)) => (base.oid != side.oid || base.mode != side.mode).then(|| Change::Modification {
            previous_entry_mode: base.mode,
            previous_oid: base.oid.to_owned(),
            entry_mode: side.mode,
            oid: side.oid.to_owned(),
        }),
        (Some(base), None) => {
            if base.mode.is_tree() {
                child_relation = Some(relation.unwrap_or_else(|| {
                    *change_id += 1;
                    Relation::Parent(*change_id)
                }));
            }
            Some(Change::Deletion {
                entry_mode: base.mode,
                oid: base.oid.to_owned(),
                relation: if base.mode.is_tree() { child_relation } else { relation },
            })
        }
        (None, Some(side)) => {
            if side.mode.is_tree() {
                child_relation = Some(relation.unwrap_or_else(|| {
                    *change_id += 1;
                    Relation::Parent(*change_id)
                }));
            }
            Some(Change::Addition {
                entry_mode: side.mode,
                oid: side.oid.to_owned(),
                relation: if side.mode.is_tree() { child_relation } else { relation },
            })
        }
        (None, None) => None,
    };
    if let Some(change) = change {
        changes.push((change, location.to_owned()));
    }
    child_relation.map(|relation| match relation {
        Relation::Parent(id) => Relation::ChildOfParent(id),
        Relation::ChildOfParent(id) => Relation::ChildOfParent(id),
    })
}

fn finish_changes(
    raw: Vec<(gix_diff::tree::visit::Change, BString)>,
    rewrites: Option<gix_diff::Rewrites>,
    source_tree: Option<&[(gix_diff::tree::visit::Change, BString)]>,
    diff_resource_cache: &mut gix_diff::blob::Platform,
    objects: &impl gix_object::FindObjectOrHeader,
) -> Result<ChangeList> {
    use gix_diff::rewrites::tracker::Change as _;
    use gix_diff::tree_with_rewrites::ChangeRef;
    let mut changes = Vec::new();
    let Some(rewrites) = rewrites else {
        for (change, location) in raw {
            track(raw_change_ref(&change, location.as_bstr()), &mut changes);
        }
        return Ok(changes);
    };

    let mut tracker = gix_diff::rewrites::Tracker::new(rewrites);
    for (change, location) in raw {
        if let Some(change) = tracker.try_push_change(change, location.as_bstr()) {
            track(raw_change_ref(&change, location.as_bstr()), &mut changes);
        }
    }
    tracker
        .emit(
            |dest, source| {
                let (id, entry_mode) = gix_diff::rewrites::tracker::Change::id_and_entry_mode(&dest.change);
                let change = if let Some(source) = source {
                    ChangeRef::Rewrite {
                        source_location: source.location,
                        source_entry_mode: source.entry_mode,
                        source_id: source.id,
                        source_relation: source.change.relation(),
                        entry_mode,
                        id: id.to_owned(),
                        relation: dest.change.relation(),
                        diff: source.diff,
                        location: dest.location,
                        copy: source.kind == gix_diff::rewrites::tracker::visit::SourceKind::Copy,
                    }
                } else {
                    raw_change_ref(&dest.change, dest.location)
                };
                track(change, &mut changes);
                std::ops::ControlFlow::Continue(())
            },
            diff_resource_cache,
            objects,
            |push| {
                if let Some(source_tree) = source_tree {
                    for (change, location) in source_tree {
                        push(change.clone(), location.as_bstr());
                    }
                }
                Ok::<_, gix_error::Error>(())
            },
        )
        .or_raise(|| message("Tree merge failed"))?;
    Ok(changes)
}

fn raw_change_ref<'a>(
    change: &'a gix_diff::tree::visit::Change,
    location: &'a bstr::BStr,
) -> gix_diff::tree_with_rewrites::ChangeRef<'a> {
    use gix_diff::tree::visit::Change::*;
    match change {
        Addition {
            entry_mode,
            oid,
            relation,
        } => gix_diff::tree_with_rewrites::ChangeRef::Addition {
            location,
            entry_mode: *entry_mode,
            relation: *relation,
            id: *oid,
        },
        Deletion {
            entry_mode,
            oid,
            relation,
        } => gix_diff::tree_with_rewrites::ChangeRef::Deletion {
            location,
            entry_mode: *entry_mode,
            relation: *relation,
            id: *oid,
        },
        Modification {
            previous_entry_mode,
            previous_oid,
            entry_mode,
            oid,
        } => gix_diff::tree_with_rewrites::ChangeRef::Modification {
            location,
            previous_entry_mode: *previous_entry_mode,
            previous_id: *previous_oid,
            entry_mode: *entry_mode,
            id: *oid,
        },
    }
}

fn source_tree_changes(
    base_tree: &gix_hash::oid,
    objects: &impl gix_object::FindObjectOrHeader,
) -> Result<Vec<(gix_diff::tree::visit::Change, BString)>> {
    fn visit(
        tree_id: &gix_hash::oid,
        path: &mut BString,
        objects: &impl gix_object::FindObjectOrHeader,
        out: &mut Vec<(gix_diff::tree::visit::Change, BString)>,
    ) -> Result<()> {
        let entries = {
            let mut buf = Vec::new();
            objects
                .find_tree_iter(tree_id, &mut buf)
                .or_raise(|| message("Tree merge failed"))?
                .map(|entry| entry.map(|entry| (entry.mode, entry.filename.to_owned(), entry.oid.to_owned())))
                .collect::<std::result::Result<Vec<_>, _>>()
                .or_raise(|| message("Tree merge failed"))?
        };
        for (mode, filename, oid) in entries {
            let old_len = path.len();
            if !path.is_empty() {
                path.push(b'/');
            }
            path.push_str(&filename);
            if mode.is_tree() {
                visit(&oid, path, objects, out)?;
            } else if mode.is_blob() {
                out.push((
                    gix_diff::tree::visit::Change::Modification {
                        previous_entry_mode: mode,
                        previous_oid: gix_hash::ObjectId::null(oid.kind()),
                        entry_mode: mode,
                        oid,
                    },
                    path.clone(),
                ));
            }
            path.truncate(old_len);
        }
        Ok(())
    }

    let mut out = Vec::new();
    visit(base_tree, &mut BString::default(), objects, &mut out)?;
    Ok(out)
}

/// Find an eligible change on our indexed side that structurally interacts with `theirs`.
///
/// `rewritten_location` belongs to a deferred change whose path passed through a directory
/// rewrite on our side. For example, if our side renamed `a` to `b`, their change at `a/x`
/// must be retried at `b/x`. The tuple contains that effective path and the index of the
/// directory rewrite which produced it, so matching starts where the change will actually
/// be applied. A rewrite is also checked at its destination, but only for a rewrite on our
/// side, to expose rewrite/rewrite interactions which are not visible at the source path.
///
/// `needs_tree_insertion` identifies a clone appended for a later retry. The resolver inserts
/// such a deferred change into its own side's [`TreeNodes`] immediately before processing it,
/// keeping future work from seeing it prematurely. Its inner index, when present, identifies
/// the opposite-side change which caused the deferral; that candidate is ignored on retry so
/// the same pair cannot defer or resolve each other again.
///
/// Changes with identical tree effects need no conflict resolution. Relations and rewrite diff
/// statistics are local bookkeeping from two independent diff operations and don't affect the
/// result. For a deletion, applying `theirs` removes the same base entry our deletion would
/// remove. An identical rewrite additionally creates the same destination. Marking our deletion
/// or rewrite applied records that one editor update represents both sides and prevents a later
/// scheduling pass from removing the shared source after descendant entries have been added there.
///
/// Applied deletions are otherwise ignored because their indexed path remains available for
/// structural lookup even though the entry itself has already been removed from the editor.
/// A deletion processed without application remains eligible: conflict resolution may have
/// retained the ancestor entry, so it can still block a tree or descendant at that path.
pub(super) fn matching(
    theirs: &Change,
    needs_tree_insertion: Option<Option<usize>>,
    rewritten_location: Option<&(BString, usize)>,
    our_tree: &TreeNodes,
    our_changes: &mut ChangeList,
) -> Option<PossibleConflict> {
    let candidate = our_tree
        .check_conflict(rewritten_location.map_or_else(|| theirs.source_location(), |(location, _)| location.as_bstr()))
        .or_else(|| match theirs {
            Change::Rewrite { location, .. } => our_tree.check_conflict(location.as_bstr()).filter(|candidate| {
                candidate
                    .change_idx()
                    .is_some_and(|idx| matches!(our_changes[idx].inner, Change::Rewrite { .. }))
            }),
            _ => None,
        });

    candidate.filter(|ours| {
        ours.change_idx()
            .zip(needs_tree_insertion.flatten())
            .is_none_or(|(ours_idx, ignore_idx)| ours_idx != ignore_idx)
            && ours.change_idx().is_none_or(|ours_idx| {
                let ours = &mut our_changes[ours_idx];
                if same_effect(&ours.inner, theirs) {
                    // Applying `theirs` also consumes an identical source removal, which must not
                    // run again after descendants have been added.
                    if matches!(theirs, Change::Deletion { .. } | Change::Rewrite { .. }) {
                        ours.mark_applied();
                    }
                    false
                } else {
                    !(ours.was_applied() && matches!(ours.inner, Change::Deletion { .. }))
                }
            })
    })
}

fn same_effect(ours: &Change, theirs: &Change) -> bool {
    std::mem::discriminant(ours) == std::mem::discriminant(theirs)
        && ours.source_location() == theirs.source_location()
        && ours.source_entry_mode_and_id() == theirs.source_entry_mode_and_id()
        && ours.location() == theirs.location()
        && ours.entry_mode_and_id() == theirs.entry_mode_and_id()
        && match (ours, theirs) {
            (Change::Rewrite { copy: ours, .. }, Change::Rewrite { copy: theirs, .. }) => ours == theirs,
            _ => true,
        }
}

/// Turn a path-index match, `candidate`, from our side into a concrete change pair for the resolution matrix.
///
/// The first tuple element indexes `our_changes`; `None` means the overlap has no eligible change to pair with. The
/// second describes a tree/non-tree boundary: [`MatchKind::EraseTree`] replaces our tree with their leaf, while
/// [`MatchKind::EraseLeaf`] replaces our leaf with their tree. An exact-path match needs no `MatchKind`.
///
/// A tree-to-non-tree candidate is pairable only when its indexed change can remove or replace that tree. A passed
/// rewritten directory is deliberately not paired here because the scheduler relocates it before calling this function.
pub(super) fn pair(candidate: &PossibleConflict, our_changes: &ChangeListRef) -> (Option<usize>, Option<MatchKind>) {
    match *candidate {
        PossibleConflict::TreeToNonTree { change_idx: Some(idx) }
            if matches!(
                our_changes[idx].inner,
                Change::Deletion { .. } | Change::Addition { .. } | Change::Rewrite { .. }
            ) =>
        {
            (Some(idx), Some(MatchKind::EraseTree))
        }
        PossibleConflict::NonTreeToTree { change_idx } => (change_idx, Some(MatchKind::EraseLeaf)),
        PossibleConflict::Match { change_idx } => (Some(change_idx), None),
        _ => (None, None),
    }
}
