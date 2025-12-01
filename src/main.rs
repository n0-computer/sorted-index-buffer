use std::collections::BTreeMap;

#[derive(Debug)]
struct PacketBuffer<T> {
    /// The underlying data buffer. Size is a power of two, and not 2.
    data: Vec<Option<T>>,
    /// The minimum valid index (inclusive).
    min: u64,
    /// The maximum valid index (exclusive, so we can model the empty buffer).
    max: u64,
}

impl Default for PacketBuffer<u64> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PacketBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        let data = Vec::with_capacity(capacity);
        Self {
            data,
            min: 0,
            max: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn buf(&self) -> &[Option<T>] {
        &self.data
    }

    fn buf_mut(&mut self) -> &mut [Option<T>] {
        &mut self.data
    }

    pub fn keys(&self) -> impl Iterator<Item = u64> + '_ {
        (self.min..self.max).filter_map(move |i| self.get(i).map(|_| i))
    }

    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        (self.min..self.max).filter_map(move |i| self.get(i))
    }

    pub fn iter(&self) -> impl Iterator<Item = (u64, &T)> + '_ {
        (self.min..self.max).filter_map(move |i| self.get(i).map(|v| (i, v)))
    }

    pub fn get(&self, index: u64) -> Option<&T> {
        if index < self.min || index >= self.max {
            return None;
        }
        let base = base(self.min, self.buf().len());
        let offset = (index - base) as usize;
        self.data[offset].as_ref()
    }

    pub fn insert(&mut self, index: u64, value: T) {
        let min1 = self.min.min(index);
        let max1 = self.max.max(index + 1);
        self.resize(min1, max1);
        // check that min and max are correctly updated
        debug_assert!(self.min == min1 && self.max == max1);
        // check that data size is the next power of two larger or equal to (max - min)
        debug_assert!(self.buf().len() == buf_len(self.min, self.max));
        self.insert0(index, value);
        self.check_invariants();
    }

    /// Resize the buffer to cover the range [new_min, new_max), while preserving existing
    /// elements.
    fn resize(&mut self, min1: u64, max1: u64) {
        if min1 == self.min && max1 == self.max {
            // nothing to do
            return;
        }
        if min1 == max1 {
            // resizing to empty buffer
            *self = Self::new();
            return;
        }
        let len1 = buf_len(min1, max1);
        if len1 == self.buf().len() {
            // just need to move data around within the existing buffer
            let base0 = base(self.min, self.buf().len());
            let base1 = base(min1, self.buf().len());
            if base1 < base0 {
                println!("resize shift {} {}", min1, max1);
                let shift = (base0 - base1) as usize;
                self.buf_mut().rotate_right(shift);
            } else if base1 > base0 {
                println!("resize shift {} {}", min1, max1);
                let shift = (base1 - base0) as usize;
                self.buf_mut().rotate_left(shift);
            }
            self.min = min1;
            self.max = max1;
            return;
        } else {
            let op = if len1 > self.buf().len() {
                "grow"
            } else {
                "shrink"
            };
            println!("resize {} {} {}", op, min1, max1);
            let mut this1 = Self::new_empty(min1, max1);
            // todo: memcpy is probably faster here
            for i in self.min..self.max {
                if let Some(v) = self.remove0(i) {
                    this1.insert0(i, v);
                }
            }
            *self = this1;
        }
    }

    pub fn remove(&mut self, index: u64) -> Option<T> {
        let res = self.remove0(index);
        if index == self.min {
            let mut i = self.min;
            while i < self.max && self.get(i).is_none() {
                i += 1;
            }
            self.resize(i, self.max);
        } else if index + 1 == self.max {
            let mut i = self.max;
            while i > self.min && self.get(i - 1).is_none() {
                i -= 1;
            }
            self.resize(self.min, i);
        }
        self.check_invariants();
        res
    }

    /// Insert value at index, assuming the buffer already covers that index.
    ///
    /// The resulting buffer may violate the invariants.
    fn insert0(&mut self, index: u64, value: T) {
        let base = base(self.min, self.buf().len());
        let offset = (index - base) as usize;
        self.buf_mut()[offset] = Some(value);
    }

    /// Remove value at index without resizing the buffer.
    ///
    /// The resulting buffer may violate the invariants.
    fn remove0(&mut self, index: u64) -> Option<T> {
        if index < self.min || index >= self.max {
            return None;
        }
        let base = base(self.min, self.buf().len());
        let offset = (index - base) as usize;
        self.buf_mut()[offset].take()
    }

    /// Create a new empty PacketBuffer covering the range [min, max).
    ///
    /// The returned buffer does not comply with the invariants.
    fn new_empty(min: u64, max: u64) -> Self {
        let n = buf_len(min, max);
        PacketBuffer {
            data: mk_empty(n),
            min,
            max,
        }
    }

    fn check_invariants(&self) {
        if self.is_empty() {
            // for the empty buffer, min and max must be zero
            debug_assert_eq!(self.min, 0);
            debug_assert_eq!(self.max, 0);
        } else {
            // for a non-empty buffer, elements min and max-1 must be valid
            debug_assert!(self.min < self.max);
            debug_assert!(self.get(self.min).is_some() || self.get(self.max - 1).is_some());
        }
    }
}

fn mk_empty<T>(n: usize) -> Vec<Option<T>> {
    let mut res = Vec::with_capacity(n);
    for _ in 0..n {
        res.push(None);
    }
    res
}

fn buf_len(min: u64, max: u64) -> usize {
    ((max - min).next_power_of_two() as usize) * 2
}

fn base(min: u64, buf_len: usize) -> u64 {
    debug_assert!(buf_len.is_power_of_two());
    debug_assert!(buf_len >= 2);
    let mask = (buf_len as u64) / 2 - 1;
    min & !mask
}

fn main() {
    let mut pb = PacketBuffer::<u64>::default();
    let mut reference = BTreeMap::<u64, u64>::new();
    for i in 0..100 {
        pb.insert(i, i * 10);
        reference.insert(i, i * 10);
        if i >= 10 {
            let ri = i - 10;
            let v1 = pb.remove(ri);
            let v2 = reference.remove(&ri);
            assert_eq!(v1, v2);
        }
    }
    for i in 100..110 {
        let v1 = pb.remove(i - 10);
        let v2 = reference.remove(&(i - 10));
        assert_eq!(v1, v2);
    }
    for (k, v) in reference.iter() {
        let v1 = pb.get(*k).cloned();
        assert_eq!(v1, Some(*v));
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;

    use super::*;

    /// Randomly permutes an iterator such that each element is displaced by at most `k` positions.
    pub fn lag_permute<I: Iterator>(iter: I, k: usize) -> impl Iterator<Item = I::Item> {
        let mut source = iter;
        let mut buffer = Vec::with_capacity(k + 1);
        let mut rng = rand::rng();

        std::iter::from_fn(move || {
            buffer.extend((&mut source).take(k + 1 - buffer.len()));

            if buffer.is_empty() {
                return None;
            }

            Some(buffer.swap_remove(rng.random_range(0..buffer.len())))
        })
    }

    #[test]
    fn test_basic() {
        let elements = lag_permute(0..10000, 100).collect::<Vec<_>>();
        let mut reference = BTreeMap::<u64, u64>::new();
        let mut pb = PacketBuffer::<u64>::default();
        let d = 100;
        let add = elements
            .iter()
            .map(|x| Some(*x))
            .chain(std::iter::repeat_n(None, d));
        let remove = std::iter::repeat_n(None, d).chain(elements.iter().map(|x| Some(*x)));
        for (a, r) in add.zip(remove) {
            if let Some(i) = a {
                pb.insert(i, i * 10);
                reference.insert(i, i * 10);
            }
            if let Some(i) = r {
                let v1 = pb.remove(i);
                let v2 = reference.remove(&i);
                assert_eq!(v1, v2);
            }
            assert_same(pb.iter(), reference.iter().map(|(k, v)| (*k, v)));
        }
    }

    fn assert_same<I1, I2, T>(iter1: I1, iter2: I2)
    where
        I1: Iterator<Item = T>,
        I2: Iterator<Item = T>,
        T: PartialEq + std::fmt::Debug,
    {
        let vec1: Vec<T> = iter1.collect();
        let vec2: Vec<T> = iter2.collect();
        assert_eq!(vec1, vec2);
    }
}
