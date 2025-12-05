/// Imagine macro parameters, but more like those Russian dolls.
///
/// Calls m!(), m!(A 1), m!(A 1, B 2), and m!(A 1, B 2, C 3) for i.e. (m, A 1, B 2, C 3)
/// where m is any macro, for any number of parameters.
macro_rules! impl_all_tuples {
    // Entry point: We need at least two items (A and B) to start the sequence
    (
        $callback:ident,
        $first_name:ident $first_idx:tt,
        $second_name:ident $second_idx:tt
        $(, $rest_name:ident $rest_idx:tt)*
    ) => {
        // 1. Generate for the first pairs (A 0, B 1) (A 0)
        $callback!($first_name $first_idx, $second_name $second_idx);
        $callback!($first_name $first_idx);

        // 2. Start the recursion with the first pair as the "base"
        impl_all_tuples!(
            @recurse
            $callback,
            ($first_name $first_idx, $second_name $second_idx) // Accumulator
            $(, $rest_name $rest_idx)* // Remaining items
        );
    };

    // Recursive Step: We have an accumulator and at least one new item to add
    (
        @recurse
        $callback:ident,
        ($($acc:tt)*), // Everything done so far
        $next_name:ident $next_idx:tt // The next item to process
        $(, $tail_name:ident $tail_idx:tt)* // Items left after this one
    ) => {
        // 1. Generate the callback with the Accumulated items + Next item
        $callback!($($acc)*, $next_name $next_idx);

        // 2. Recurse, moving "Next" into the accumulator
        impl_all_tuples!(
            @recurse
            $callback,
            ($($acc)*, $next_name $next_idx)
            $(, $tail_name $tail_idx)*
        );
    };

    // Base Case: Recursion ends when there are no "next" items left
    (@recurse $callback:ident, ($($acc:tt)*) $(,)?) => {
        // Done.
    };
}

macro_rules! impl_component {
    ($($t:ty),*) => {
        $(
            impl crate::component::Component for $t {
                #[inline(always)]
                fn get_id() -> crate::component::ComponentId {
                    use std::sync::atomic::{AtomicUsize, Ordering};
                    // Unique static per type instantiation
                    static CACHED_ID: AtomicUsize = AtomicUsize::new(usize::MAX);

                    let id = CACHED_ID.load(Ordering::Relaxed);
                    if id != usize::MAX {
                        return crate::component::ComponentId(id);
                    }

                    // Fallback to global allocator
                    let new_id = crate::component::allocate_component_id::<Self>();
                    CACHED_ID.store(new_id.0, Ordering::Relaxed);
                    new_id
                }
            }
        )*
    };
}

macro_rules! impl_component_with_lifetime {
    ($($t:ty),*) => {
        $(
            impl<'a> crate::component::Component for $t {
                #[inline(always)]
                fn get_id() -> crate::component::ComponentId {
                    use std::sync::atomic::{AtomicUsize, Ordering};
                    // Unique static per type instantiation
                    static CACHED_ID: AtomicUsize = AtomicUsize::new(usize::MAX);

                    let id = CACHED_ID.load(Ordering::Relaxed);
                    if id != usize::MAX {
                        return crate::component::ComponentId(id);
                    }

                    // Fallback to global allocator
                    let new_id = crate::component::allocate_component_id::<Self>();
                    CACHED_ID.store(new_id.0, Ordering::Relaxed);
                    new_id
                }
            }
        )*
    };
}
