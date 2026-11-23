//! Benchmarks for contiguous and sparse index patterns.

use std::{hint::black_box, ops::Bound};

use criterion::{criterion_group, criterion_main, Criterion};
use sorted_index_buffer::SortedIndexBuffer;

/// A 104-byte value, since slot size drives the cost of every scenario.
type Entry = [u64; 13];

fn entry(size: u64) -> Entry {
    let mut entry = [0; 13];
    entry[0] = size;
    entry
}

fn has_sized_entry(entries: &SortedIndexBuffer<Entry>) -> bool {
    entries.values().any(|entry| entry[0] != 0)
}

/// Inserts windows of 2048 indices and removes all but two entries of each window.
fn sliding_window(c: &mut Criterion) {
    const WINDOW: u64 = 2048;
    const ROUNDS: u64 = 128;

    c.bench_function("sliding_window", |b| {
        b.iter(|| {
            let mut entries = SortedIndexBuffer::default();
            for round in 0..ROUNDS {
                let base = round * WINDOW;
                for offset in 0..WINDOW {
                    let size = if offset == 3 { 0 } else { 1200 };
                    entries.insert(base + offset, entry(size));
                }

                let largest = base + WINDOW - 2;
                let kept = base + WINDOW / 2;
                black_box(entries.get(largest));
                for (index, _) in entries.iter_range(0..largest) {
                    black_box(index);
                }

                let stale: Vec<u64> = entries.keys_range(0..base).collect();
                for index in stale {
                    black_box(entries.remove(index));
                }

                let done: Vec<u64> = entries
                    .keys_range(base..=largest)
                    .filter(|index| *index != kept)
                    .collect();
                for index in done {
                    black_box(entries.remove(index));
                }

                for (_, entry) in entries.iter_range(..).take(2) {
                    black_box(entry);
                }
                black_box(has_sized_entry(&entries));
            }

            black_box(entries.get(ROUNDS * WINDOW - 1));
            for entry in entries.into_iter() {
                black_box(entry);
            }
        })
    });
}

/// Keeps the oldest entries while a tail of 1001 entries moves past them.
fn pinned_oldest_entry(c: &mut Criterion) {
    const PINNED: u64 = 50;
    const MAX_TAIL: u64 = 1_000;
    const INSERTS: u64 = 20_000;

    c.bench_function("pinned_oldest_entry", |b| {
        b.iter(|| {
            let mut entries = SortedIndexBuffer::default();
            for index in 0..PINNED {
                entries.insert(index, entry(1200));
            }
            let newest_pinned = PINNED - 1;
            let mut tail = 0;

            for i in 0..INSERTS {
                if tail > MAX_TAIL {
                    let oldest = entries
                        .keys_range((Bound::Excluded(newest_pinned), Bound::Unbounded))
                        .next()
                        .expect("tail entries are kept");
                    black_box(entries.remove(oldest));
                } else {
                    tail += 1;
                }
                entries.insert(PINNED + i, entry(0));

                if i % 256 == 255 {
                    black_box(has_sized_entry(&entries));
                    black_box(entries.iter_range(..).next());
                    for (index, _) in entries.iter_range(0..PINNED) {
                        black_box(index);
                    }
                }
            }

            black_box(
                entries
                    .keys_range((Bound::Excluded(newest_pinned), Bound::Unbounded))
                    .next(),
            );
        })
    });
}

/// Pins one more entry after every 2000 inserts, leaving a hole before each.
fn pinned_holes(c: &mut Criterion) {
    const PINNED: u64 = 50;
    const MAX_TAIL: u64 = 1_000;
    const CYCLES: u64 = 8;
    const INSERTS_PER_CYCLE: u64 = 2_000;

    c.bench_function("pinned_holes", |b| {
        b.iter(|| {
            let mut entries = SortedIndexBuffer::default();
            for index in 0..PINNED {
                entries.insert(index, entry(1200));
            }
            let mut newest_pinned = PINNED - 1;
            let mut index = PINNED;

            for _ in 0..CYCLES {
                let mut tail = 0;
                for _ in 0..INSERTS_PER_CYCLE {
                    if tail > MAX_TAIL {
                        let oldest = entries
                            .keys_range((Bound::Excluded(newest_pinned), Bound::Unbounded))
                            .next()
                            .expect("tail entries are kept");
                        black_box(entries.remove(oldest));
                    } else {
                        tail += 1;
                    }
                    entries.insert(index, entry(0));
                    index += 1;
                }

                entries.insert(index, entry(1200));
                newest_pinned = index;
                index += 1;

                black_box(has_sized_entry(&entries));
                black_box(entries.iter_range(..).next());
                for (index, _) in entries.iter_range(0..PINNED) {
                    black_box(index);
                }
            }

            black_box(has_sized_entry(&entries));
        })
    });
}

criterion_group!(benches, sliding_window, pinned_oldest_entry, pinned_holes);
criterion_main!(benches);
