use std::{
    collections::BTreeMap,
    ops::Bound::{self, Excluded, Included, Unbounded},
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
};

use proptest::prelude::*;

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
fn test_usage() {
    let elements = lag_permute(0..10000, 100).collect::<Vec<_>>();
    let mut reference = BTreeMap::<u64, u64>::new();
    let mut pb = SortedIndexBuffer::<u64>::default();
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
        pb.check_invariants_expensive();
    }
    assert!(reference.is_empty());
    assert!(pb.is_empty());
}

#[test]
fn test_range_iterators() {
    let mut pb = SortedIndexBuffer::default();
    for i in 0..100 {
        pb.insert(i, i * 10);
    }

    // Test keys_range
    let keys: Vec<_> = pb.keys_range(10..20).collect();
    assert_eq!(keys, (10..20).collect::<Vec<_>>());

    // Test reverse
    let keys_rev: Vec<_> = pb.keys_range(10..20).rev().collect();
    assert_eq!(keys_rev, (10..20).rev().collect::<Vec<_>>());

    // Test values_range
    let values: Vec<_> = pb.values_range(10..20).cloned().collect();
    assert_eq!(values, (10..20).map(|i| i * 10).collect::<Vec<_>>());

    // Test iter_range
    let pairs: Vec<_> = pb.iter_range(10..20).map(|(k, v)| (k, *v)).collect();
    assert_eq!(pairs, (10..20).map(|i| (i, i * 10)).collect::<Vec<_>>());
}

#[test]
fn test_retain() {
    let mut pb = SortedIndexBuffer::default();
    for i in 0..100 {
        pb.insert(i, i * 10);
    }

    pb.retain(|i, _v| i % 2 == 0);

    assert_eq!(pb.keys().next(), Some(0));
    assert_eq!(pb.keys().next_back(), Some(98));
    assert_eq!(pb.keys().count(), 50);
    for i in 0..100 {
        if i % 2 == 0 {
            assert_eq!(pb.get(i), Some(&(i * 10)));
        } else {
            assert_eq!(pb.get(i), None);
        }
    }
    pb.check_invariants_expensive();

    pb.retain(|_, _| false);
    assert!(pb.is_empty());
    pb.check_invariants_expensive();
}

#[test]
fn test_retain_range() {
    let mut pb = SortedIndexBuffer::default();
    for i in 0..100 {
        pb.insert(i, i * 10);
    }

    pb.retain_range(20..80);

    assert_eq!(pb.keys().next(), Some(20));
    assert_eq!(pb.keys().next_back(), Some(79));
    assert_eq!(pb.keys().count(), 60);
    for i in 20..80 {
        assert_eq!(pb.get(i), Some(&(i * 10)));
    }
    pb.check_invariants_expensive();

    // Retain with range outside current bounds -> empty
    pb.retain_range(200..300);
    assert!(pb.is_empty());
    pb.check_invariants_expensive();

    // Rebuild and retain with superset range -> no-op
    for i in 10..20 {
        pb.insert(i, i * 10);
    }
    pb.retain_range(0..100);
    assert_eq!(pb.keys().count(), 10);
    pb.check_invariants_expensive();
}

#[test]
fn test_pinned_entry_does_not_grow_capacity() {
    const TAIL: u64 = 1000;
    let mut pb = SortedIndexBuffer::default();
    pb.insert(0, 0);
    for index in 1..100_000 {
        if index > TAIL {
            let oldest = pb.keys_range(1..).next().unwrap();
            pb.remove(oldest);
        }
        pb.insert(index, index);
    }
    assert_eq!(pb.keys().count(), 1 + TAIL as usize);
    assert_capacity_bound(&pb);
}

#[test]
fn test_pinned_holes_do_not_grow_capacity() {
    const TAIL: u64 = 1000;
    let mut pb = SortedIndexBuffer::default();
    for index in 0..50 {
        pb.insert(index, index);
    }
    let mut pinned = 49;
    let mut index = 50;
    for _ in 0..20 {
        let mut tail = 0;
        for _ in 0..10 * TAIL {
            if tail == TAIL {
                let oldest = pb.keys_range(pinned + 1..).next().unwrap();
                pb.remove(oldest);
            } else {
                tail += 1;
            }
            pb.insert(index, index);
            index += 1;
        }
        pb.insert(index, index);
        pinned = index;
        index += 1;
    }
    assert_capacity_bound(&pb);
}

#[test]
fn test_retain_panic_leaves_buffer_consistent() {
    for panic_chunk in 1..8u64 {
        for keep_mask in 0..256u64 {
            let mut pb = SortedIndexBuffer::default();
            for chunk in 0..8u64 {
                pb.insert(chunk * 64, chunk);
            }
            let result = catch_unwind(AssertUnwindSafe(|| {
                pb.retain(|k, _| {
                    if k / 64 == panic_chunk {
                        resume_unwind(Box::new(()));
                    }
                    keep_mask & (1 << (k / 64)) != 0
                })
            }));
            assert!(result.is_err());

            let keys: Vec<u64> = pb.keys().collect();
            for &k in &keys {
                assert!(pb.get(k).is_some());
            }
            for start in 0..8u64 {
                for end in start..8u64 {
                    let range = start * 64..=end * 64;
                    let expected = keys.iter().filter(|k| range.contains(k)).count();
                    assert_eq!(pb.iter_range(range).count(), expected);
                }
            }
        }
    }
}

fn assert_capacity_bound<T>(pb: &SortedIndexBuffer<T>) {
    let entries = pb.keys().count();
    let gaps = pb
        .keys()
        .zip(pb.keys().skip(1))
        .filter(|(a, b)| b - a > 1)
        .count();
    let runs = if entries == 0 { 0 } else { gaps + 1 };
    assert!(pb.capacity() <= CHUNK_SIZE * entries);
    assert!(pb.capacity() <= entries + 2 * CHUNK_SIZE * runs);
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

#[derive(Debug, Clone)]
enum InsertRemoveGetOp {
    Insert(u64, u64),
    Remove(u64),
    Get(u64),
}

fn key_strategy() -> impl Strategy<Value = u64> {
    prop_oneof![
        0..1000u64,
        (0..50u64).prop_map(|k| k * 1000),
        (u64::MAX - 200)..=u64::MAX,
    ]
}

fn op_strategy() -> impl Strategy<Value = InsertRemoveGetOp> {
    prop_oneof![
        (key_strategy(), any::<u64>()).prop_map(|(k, v)| InsertRemoveGetOp::Insert(k, v)),
        key_strategy().prop_map(InsertRemoveGetOp::Remove),
        key_strategy().prop_map(InsertRemoveGetOp::Get),
    ]
}

#[test_strategy::proptest]
fn test_insert_remove_get(
    #[strategy(prop::collection::vec(op_strategy(), 0..1000))] ops: Vec<InsertRemoveGetOp>,
) {
    let mut pb = SortedIndexBuffer::default();
    let mut reference = BTreeMap::new();

    for op in ops {
        match op {
            InsertRemoveGetOp::Insert(k, v) => {
                pb.insert(k, v);
                reference.insert(k, v);
            }
            InsertRemoveGetOp::Remove(k) => {
                let v1 = pb.remove(k);
                let v2 = reference.remove(&k);
                assert_eq!(v1, v2);
            }
            InsertRemoveGetOp::Get(k) => {
                let v1 = pb.get(k);
                let v2 = reference.get(&k);
                assert_eq!(v1, v2);
            }
        }
        pb.check_invariants_expensive();
    }

    // Final state should match
    assert_same(pb.iter(), reference.iter().map(|(k, v)| (*k, v)));
    assert_capacity_bound(&pb);
}

#[test_strategy::proptest]
fn test_range_ops(
    #[strategy(prop::collection::vec((key_strategy(), any::<u64>()), 0..300))] entries: Vec<(
        u64,
        u64,
    )>,
    #[strategy(key_strategy())] a: u64,
    #[strategy(key_strategy())] b: u64,
) {
    let (lo, hi) = (a.min(b), a.max(b));
    let mut pb = SortedIndexBuffer::default();
    let mut reference = BTreeMap::new();
    for (k, v) in entries {
        pb.insert(k, v);
        reference.insert(k, v);
    }

    let ranges: [(Bound<u64>, Bound<u64>); 5] = [
        (Included(lo), Included(hi)),
        (Included(lo), Excluded(hi)),
        (Excluded(lo), Unbounded),
        (Unbounded, Included(hi)),
        (Unbounded, Unbounded),
    ];
    for range in ranges {
        let expected = reference.range(range).map(|(k, v)| (*k, v));
        assert_same(pb.iter_range(range), expected.clone());
        assert_same(pb.iter_range(range).rev(), expected.clone().rev());

        let mut retained = pb.clone();
        retained.retain_range(range);
        retained.check_invariants_expensive();
        assert_same(retained.iter(), expected);
    }

    for v in pb.values_range_mut(lo..=hi) {
        *v = v.wrapping_add(1);
    }
    for (_, v) in reference.range_mut(lo..=hi) {
        *v = v.wrapping_add(1);
    }
    pb.retain(|k, v| (k ^ *v) % 3 != 0);
    reference.retain(|k, v| (k ^ *v) % 3 != 0);
    pb.check_invariants_expensive();
    assert_capacity_bound(&pb);
    assert_same(pb.into_iter().rev(), reference.into_iter().rev());
}
