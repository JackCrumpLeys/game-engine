use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Index, parse_macro_input};

// Register the "interpolate" helper attribute here so the compiler doesn't complain
#[proc_macro_derive(Interpolate, attributes(interpolate))]
pub fn derive_interpolate(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let logic = match input.data {
        Data::Struct(data) => expand_struct_fields(&data.fields),
        _ => {
            return syn::Error::new_spanned(name, "Interpolate can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let expanded = quote! {
        impl #impl_generics Interpolate for #name #ty_generics #where_clause {
            fn interpolate(&self, other: &Self, factor: f32) -> Self {
                #logic
            }
        }
    };

    TokenStream::from(expanded)
}

fn expand_struct_fields(fields: &Fields) -> proc_macro2::TokenStream {
    // 1. Check if ANY field in the struct has the #[interpolate] attribute.
    // If yes, we enter "Explicit Mode" (only interpolate marked fields).
    // If no, we enter "Default Mode" (interpolate everything).
    let has_explicit_attr = match fields {
        Fields::Named(f) => f
            .named
            .iter()
            .any(|field| has_interpolate_attr(&field.attrs)),
        Fields::Unnamed(f) => f
            .unnamed
            .iter()
            .any(|field| has_interpolate_attr(&field.attrs)),
        Fields::Unit => false,
    };

    match fields {
        // Case: Named Struct { x: ..., y: ... }
        Fields::Named(named_fields) => {
            let recurse = named_fields.named.iter().map(|f| {
                let name = &f.ident;
                let is_marked = has_interpolate_attr(&f.attrs);

                // Logic:
                // If we are in Explicit Mode, only interpolate if marked.
                // If we are in Default Mode, interpolate everything.
                let should_interpolate = if has_explicit_attr { is_marked } else { true };

                if should_interpolate {
                    // Perform interpolation
                    quote! { #name: self.#name.interpolate(&other.#name, factor) }
                } else {
                    // Just copy the target value.
                    // We use 'other' (the new state) so that properties snap instantly
                    // to the new value rather than getting stuck on the old value.
                    quote! { #name: other.#name.clone() }
                }
            });
            quote! { Self { #(#recurse),* } }
        }

        // Case: Tuple Struct (f32, f32)
        Fields::Unnamed(unnamed_fields) => {
            let recurse = unnamed_fields.unnamed.iter().enumerate().map(|(i, f)| {
                let index = Index::from(i);
                let is_marked = has_interpolate_attr(&f.attrs);

                let should_interpolate = if has_explicit_attr { is_marked } else { true };

                if should_interpolate {
                    quote! { self.#index.interpolate(&other.#index, factor) }
                } else {
                    quote! { other.#index.clone() }
                }
            });
            quote! { Self ( #(#recurse),* ) }
        }

        // Case: Unit Struct
        Fields::Unit => quote! { Self },
    }
}

#[proc_macro_derive(Component)]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let expanded = quote! {
        impl game_engine_ecs::component::Component for #name {
            #[inline(always)]
            fn get_id() -> game_engine_ecs::component::ComponentId {
                use std::sync::atomic::{AtomicUsize, Ordering};
                // Unique static per type instantiation
                static CACHED_ID: AtomicUsize = AtomicUsize::new(usize::MAX);

                let id = CACHED_ID.load(Ordering::Relaxed);
                if id != usize::MAX {
                    return game_engine_ecs::component::ComponentId(id);
                }

                // Fallback to global allocator
                let new_id = game_engine_ecs::component::allocate_component_id::<Self>();
                CACHED_ID.store(new_id.0, Ordering::Relaxed);
                new_id
            }
        }
    };

    TokenStream::from(expanded)
}

/// Helper to check if the #[interpolate] attribute is present
fn has_interpolate_attr(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| attr.path().is_ident("interpolate"))
}
