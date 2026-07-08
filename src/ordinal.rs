use crate::{tconorm, tnorm, TConormFamily};

/// One interval of an ordinal-sum t-norm or t-conorm.
///
/// When both inputs fall inside `[lower, upper]`, they are rescaled into
/// `[0, 1]`, evaluated with `family`, and rescaled back into the segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrdinalSegment {
    /// Inclusive lower interval endpoint.
    pub lower: f64,
    /// Inclusive upper interval endpoint.
    pub upper: f64,
    /// Family used inside the interval.
    pub family: TConormFamily,
}

impl OrdinalSegment {
    /// Create a segment.
    pub const fn new(lower: f64, upper: f64, family: TConormFamily) -> Self {
        Self {
            lower,
            upper,
            family,
        }
    }

    fn contains(self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }

    fn is_valid(self) -> bool {
        0.0 <= self.lower && self.lower < self.upper && self.upper <= 1.0
    }
}

/// Ordinal sum over disjoint sub-intervals of `[0, 1]`.
///
/// Outside configured segments, t-norm evaluation falls back to minimum and
/// t-conorm evaluation falls back to maximum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrdinalSum<'a> {
    segments: &'a [OrdinalSegment],
}

impl<'a> OrdinalSum<'a> {
    /// Create an ordinal sum from non-overlapping, ascending segments.
    pub fn new(segments: &'a [OrdinalSegment]) -> Option<Self> {
        let mut previous_upper = 0.0;
        for (index, segment) in segments.iter().copied().enumerate() {
            if !segment.is_valid() {
                return None;
            }
            if index > 0 && segment.lower < previous_upper {
                return None;
            }
            previous_upper = segment.upper;
        }
        Some(Self { segments })
    }

    /// Borrow the validated segments.
    pub const fn segments(self) -> &'a [OrdinalSegment] {
        self.segments
    }

    /// Evaluate the ordinal-sum t-norm.
    pub fn tnorm(self, a: f64, b: f64) -> f64 {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        for segment in self.segments.iter().copied() {
            if segment.contains(a) && segment.contains(b) {
                let width = segment.upper - segment.lower;
                let local_a = (a - segment.lower) / width;
                let local_b = (b - segment.lower) / width;
                return segment.lower + width * tnorm(segment.family, local_a, local_b);
            }
        }
        a.min(b)
    }

    /// Evaluate the ordinal-sum t-conorm.
    pub fn tconorm(self, a: f64, b: f64) -> f64 {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        for segment in self.segments.iter().copied() {
            if segment.contains(a) && segment.contains(b) {
                let width = segment.upper - segment.lower;
                let local_a = (a - segment.lower) / width;
                let local_b = (b - segment.lower) / width;
                return segment.lower + width * tconorm(segment.family, local_a, local_b);
            }
        }
        a.max(b)
    }
}
