// (Thx bevy) https://docs.rs/bevy_platform/0.17.3/src/bevy_platform/hash.rs.html#148
pub mod graph;
use std::hash::Hash;
use std::{
    any::Any,
    hash::{BuildHasher, Hasher},
};

/// [`BuildHasher`] for types that already contain a high-quality hash.
#[derive(Clone, Default)]
pub struct NoOpHash;

impl BuildHasher for NoOpHash {
    type Hasher = NoOpHasher;

    fn build_hasher(&self) -> Self::Hasher {
        NoOpHasher(0)
    }
}

#[doc(hidden)]
pub struct NoOpHasher(u64);

// This is for types that already contain a high-quality hash and want to skip
// re-hashing that hash.
impl Hasher for NoOpHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // This should never be called by consumers. Prefer to call `write_u64` instead.
        // Don't break applications (slower fallback, just check in test):
        self.0 = bytes.iter().fold(self.0, |hash, b| {
            hash.rotate_left(8).wrapping_add(*b as u64)
        });
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

/// An object safe version of [`Eq`]. This trait is automatically implemented
/// for any `'static` type that implements `Eq`.
pub trait DynEq: Any {
    /// Casts the type to `dyn Any`.
    fn as_any(&self) -> &dyn Any;

    /// This method tests for `self` and `other` values to be equal.
    ///
    /// Implementers should avoid returning `true` when the underlying types are
    /// not the same.
    fn dyn_eq(&self, other: &dyn DynEq) -> bool;
}

impl<T> DynEq for T
where
    T: Any + Eq,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dyn_eq(&self, other: &dyn DynEq) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<T>() {
            return self == other;
        }
        false
    }
}

/// An object safe version of [`Hash`]. This trait is automatically implemented
/// for any `'static` type that implements `Hash`.
pub trait DynHash: DynEq {
    /// Casts the type to `dyn Any`.
    fn as_dyn_eq(&self) -> &dyn DynEq;

    /// Feeds this value into the given [`Hasher`].
    ///
    /// [`Hasher`]: std::hash::Hasher
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

impl<T> DynHash for T
where
    T: DynEq + Hash,
{
    fn as_dyn_eq(&self) -> &dyn DynEq {
        self
    }

    fn dyn_hash(&self, mut state: &mut dyn Hasher) {
        T::hash(self, &mut state);
        self.type_id().hash(&mut state);
    }
}
