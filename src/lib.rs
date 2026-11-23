use std::collections::VecDeque;

#[cfg(test)]
mod tests;

/// The number of consecutive indices a chunk covers.
const CHUNK_SIZE: usize = 64;

const _: () = assert!(CHUNK_SIZE <= u64::BITS as usize);

/// A buffer indexed by a u64 sequence number.
///
/// This behaves idential to a BTreeMap<u64, T>, but is optimized for the case where
/// the indices are mostly contiguous, with occasional gaps.
///
/// Very sparse indices cost up to 64 slots per entry, so it should not be used as a
/// general-purpose sorted map.
///
/// # Internals
///
/// Entries are stored in chunks of 64 slots, each covering an aligned range of indices.
/// The chunks are kept in a `VecDeque` in ascending order, and a chunk is freed once it
/// holds no entries. Storage is therefore at most one slot per entry plus two chunks per
/// run of consecutive indices, whatever the gaps between the runs.
///
/// ```text
/// |_x_xxxx_|        |___xx___|   chunks covering 0..8 and 16..24, if a chunk had 8 slots
/// ```
///
/// Access, insertion and removal find their chunk in O(1) when no chunk is missing
/// between it and the first or the last chunk, and in O(log n) otherwise. Adding or
/// freeing a chunk in the middle moves up to half of the chunk list.
#[derive(Debug, Clone)]
pub struct SortedIndexBuffer<T> {
    /// Chunks holding at least one entry, in ascending order.
    chunks: VecDeque<Chunk<T>>,
}

impl<T> Default for SortedIndexBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SortedIndexBuffer<T> {
    /// Create a new SortedIndexBuffer with chunk-list room for about `capacity` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            chunks: VecDeque::with_capacity(capacity.div_ceil(CHUNK_SIZE)),
        }
    }

    /// Create a new, empty SortedIndexBuffer.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Returns true if the buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Returns the number of slots in allocated chunks.
    pub fn capacity(&self) -> usize {
        self.chunks.len() * CHUNK_SIZE
    }

    /// Returns true if the buffer contains an element at the given index.
    pub fn contains_key(&self, index: u64) -> bool {
        self.get(index).is_some()
    }

    /// Iterate over all keys in the given index range in ascending order.
    pub fn keys_range<R: std::ops::RangeBounds<u64>>(
        &self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = u64> + '_ {
        self.iter_range(range).map(|(index, _)| index)
    }

    /// Iterate over all values in the given index range in ascending order of their keys.
    pub fn values_range<R: std::ops::RangeBounds<u64>>(
        &self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = &T> + '_ {
        self.iter_range(range).map(|(_, value)| value)
    }

    /// Iterate over all values in the given index range in ascending order of their keys.
    pub fn values_range_mut<R: std::ops::RangeBounds<u64>>(
        &mut self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = &mut T> + '_ {
        let (start, end) = bounds(range);
        let chunks = self.chunk_range(start, end);
        self.chunks.range_mut(chunks).flat_map(move |chunk| {
            let (base, occupied) = (chunk.base, chunk.occupied);
            chunk
                .slots
                .iter_mut()
                .enumerate()
                .filter_map(move |(slot, value)| {
                    let index = base + slot as u64;
                    if !is_occupied(occupied, slot) || index < start || index > end {
                        return None;
                    }
                    value.as_mut()
                })
        })
    }

    /// Iterate over all (index, value) pairs in the given index range in ascending order of their keys.
    pub fn iter_range<R: std::ops::RangeBounds<u64>>(
        &self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = (u64, &T)> + '_ {
        let (start, end) = bounds(range);
        let chunks = self.chunk_range(start, end);
        self.chunks.range(chunks).flat_map(move |chunk| {
            let (base, occupied) = (chunk.base, chunk.occupied);
            chunk
                .slots
                .iter()
                .enumerate()
                .filter_map(move |(slot, value)| {
                    let index = base + slot as u64;
                    if !is_occupied(occupied, slot) || index < start || index > end {
                        return None;
                    }
                    Some((index, value.as_ref()?))
                })
        })
    }

    pub fn retain(&mut self, f: impl FnMut(u64, &mut T) -> bool) {
        let mut f = f;
        // A panic in `f` must leave the chunks in order, so empty chunks go in a second pass.
        for chunk in &mut self.chunks {
            chunk.retain(&mut f);
        }
        self.chunks.retain(|chunk| !chunk.is_empty());
        self.check_invariants();
    }

    /// Retain only the elements in the given index range.
    pub fn retain_range<R: std::ops::RangeBounds<u64>>(&mut self, range: R) {
        let (start, end) = bounds(range);
        let keep = self.chunk_range(start, end);
        self.chunks.truncate(keep.end);
        self.chunks.drain(..keep.start);
        // Only the first and the last chunk can hold indices outside the range.
        if let Some(chunk) = self.chunks.front_mut() {
            chunk.retain(|index, _| index >= start);
        }
        if let Some(chunk) = self.chunks.back_mut() {
            chunk.retain(|index, _| index <= end);
        }
        if self.chunks.front().is_some_and(Chunk::is_empty) {
            self.chunks.pop_front();
        }
        if self.chunks.back().is_some_and(Chunk::is_empty) {
            self.chunks.pop_back();
        }
        self.check_invariants();
    }

    /// Iterate over all keys in the buffer in ascending order.
    pub fn keys(&self) -> impl DoubleEndedIterator<Item = u64> + '_ {
        self.keys_range(..)
    }

    /// Iterate over all values in the buffer in ascending order of their keys.
    pub fn values(&self) -> impl DoubleEndedIterator<Item = &T> + '_ {
        self.values_range(..)
    }

    /// Iterate over all values in the buffer in ascending order of their keys.
    pub fn values_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut T> + '_ {
        self.values_range_mut(..)
    }

    /// Iterate over all (index, value) pairs in the buffer in ascending order of their keys.
    ///
    /// Values are returned by reference.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (u64, &T)> + '_ {
        self.iter_range(..)
    }

    /// Turn into an iterator over all (index, value) pairs in the buffer in ascending order of their keys.
    ///
    /// This is an explicit method instead of implementing IntoIterator, so we can return a
    /// DoubleEndedIterator without having to name the iterator type.
    #[allow(clippy::should_implement_trait)]
    pub fn into_iter(self) -> impl DoubleEndedIterator<Item = (u64, T)> {
        self.chunks.into_iter().flat_map(|chunk| {
            let base = chunk.base;
            let slots: Box<[Option<T>]> = chunk.slots;
            Vec::from(slots)
                .into_iter()
                .enumerate()
                .filter_map(move |(slot, value)| Some((base + slot as u64, value?)))
        })
    }

    /// Returns the positions of the chunks overlapping `start..=end`.
    fn chunk_range(&self, start: u64, end: u64) -> std::ops::Range<usize> {
        if start > end {
            return 0..0;
        }
        let first = match self.position(chunk_base(start)) {
            Ok(pos) | Err(pos) => pos,
        };
        let last = match self.position(chunk_base(end)) {
            Ok(pos) => pos + 1,
            Err(pos) => pos,
        };
        first..last
    }

    /// Returns the position of the chunk at `base`, or the position to insert it at.
    fn position(&self, base: u64) -> Result<usize, usize> {
        let (Some(first), Some(last)) = (self.chunks.front(), self.chunks.back()) else {
            return Err(0);
        };
        let last_pos = self.chunks.len() - 1;
        if base > last.base {
            return Err(last_pos + 1);
        }
        if base == last.base {
            return Ok(last_pos);
        }
        if base < first.base {
            return Err(0);
        }
        // Bases strictly increase, so counting chunks from either end finds the chunk
        // when no chunk is missing in between. Every guess is checked.
        let from_first = ((base - first.base) / CHUNK_SIZE as u64) as usize;
        let from_last = ((last.base - base) / CHUNK_SIZE as u64) as usize;
        for pos in [from_first, last_pos.saturating_sub(from_last)] {
            if self.chunks.get(pos).is_some_and(|chunk| chunk.base == base) {
                return Ok(pos);
            }
        }
        self.chunks.binary_search_by_key(&base, |chunk| chunk.base)
    }

    /// Get a reference to the value at the given index, if it exists.
    pub fn get(&self, index: u64) -> Option<&T> {
        let pos = self.position(chunk_base(index)).ok()?;
        self.chunks.get(pos)?.slots[slot(index)].as_ref()
    }

    /// Insert value at index.
    pub fn insert(&mut self, index: u64, value: T) {
        let base = chunk_base(index);
        let pos = match self.position(base) {
            Ok(pos) => pos,
            Err(pos) => {
                self.chunks.insert(pos, Chunk::new(base));
                pos
            }
        };
        if let Some(chunk) = self.chunks.get_mut(pos) {
            let slot = slot(index);
            chunk.slots[slot] = Some(value);
            chunk.occupied |= 1 << slot;
        }
        self.check_invariants();
    }

    /// Remove value at index.
    pub fn remove(&mut self, index: u64) -> Option<T> {
        let pos = self.position(chunk_base(index)).ok()?;
        let chunk = self.chunks.get_mut(pos)?;
        let slot = slot(index);
        chunk.occupied &= !(1 << slot);
        let value = chunk.slots[slot].take();
        if chunk.is_empty() {
            self.chunks.remove(pos);
        }
        self.check_invariants();
        value
    }

    /// Check that the invariants of the SortedIndexBuffer hold.
    ///
    /// This should be called after each public &mut method.
    ///
    /// It is a noop in release builds.
    fn check_invariants(&self) {
        // Only the ends are checked, so debug builds stay O(1) per operation.
        debug_assert!(!self.chunks.front().is_some_and(Chunk::is_empty));
        debug_assert!(!self.chunks.back().is_some_and(Chunk::is_empty));
    }

    /// Same as `check_invariants`, but checks every chunk.
    ///
    /// This is more expensive, so only used in tests.
    #[cfg(test)]
    fn check_invariants_expensive(&self) {
        self.check_invariants();
        for chunk in &self.chunks {
            debug_assert!(!chunk.is_empty());
            debug_assert_eq!(chunk.base, chunk_base(chunk.base));
            for (slot, value) in chunk.slots.iter().enumerate() {
                debug_assert_eq!(is_occupied(chunk.occupied, slot), value.is_some());
            }
        }
        for (a, b) in self.chunks.iter().zip(self.chunks.iter().skip(1)) {
            debug_assert!(a.base < b.base);
        }
    }
}

/// A single aligned range of `CHUNK_SIZE` slots.
#[derive(Debug, Clone)]
struct Chunk<T> {
    /// The index of the first slot.
    base: u64,
    /// Bit `i` is set if slot `i` holds a value.
    occupied: u64,
    slots: Box<[Option<T>; CHUNK_SIZE]>,
}

impl<T> Chunk<T> {
    fn new(base: u64) -> Self {
        Self {
            base,
            occupied: 0,
            slots: Box::new([const { None }; CHUNK_SIZE]),
        }
    }

    fn is_empty(&self) -> bool {
        self.occupied == 0
    }

    /// Removes the values for which `f` returns false.
    fn retain(&mut self, mut f: impl FnMut(u64, &mut T) -> bool) {
        for (slot, value) in self.slots.iter_mut().enumerate() {
            let Some(v) = value else {
                continue;
            };
            if !f(self.base + slot as u64, v) {
                *value = None;
                self.occupied &= !(1 << slot);
            }
        }
    }
}

/// Returns the inclusive bounds of `range`, with `start > end` if it is empty.
fn bounds<R: std::ops::RangeBounds<u64>>(range: R) -> (u64, u64) {
    use std::ops::Bound;

    let start = match range.start_bound() {
        Bound::Included(&n) => Some(n),
        Bound::Excluded(&n) => n.checked_add(1),
        Bound::Unbounded => Some(0),
    };
    let end = match range.end_bound() {
        Bound::Included(&n) => Some(n),
        Bound::Excluded(&n) => n.checked_sub(1),
        Bound::Unbounded => Some(u64::MAX),
    };
    match (start, end) {
        (Some(start), Some(end)) => (start, end),
        _ => (u64::MAX, 0),
    }
}

/// Returns the index of the first slot of the chunk covering `index`.
#[inline(always)]
fn chunk_base(index: u64) -> u64 {
    index - index % CHUNK_SIZE as u64
}

/// Returns the slot of `index` within its chunk.
#[inline(always)]
fn slot(index: u64) -> usize {
    (index % CHUNK_SIZE as u64) as usize
}

#[inline(always)]
fn is_occupied(occupied: u64, slot: usize) -> bool {
    (occupied >> slot) & 1 == 1
}
