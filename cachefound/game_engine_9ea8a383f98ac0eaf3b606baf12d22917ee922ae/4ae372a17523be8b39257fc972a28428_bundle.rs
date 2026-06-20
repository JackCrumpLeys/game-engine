»use crate::component::{Component, ComponentId, ComponentMeta};
use crate::storage::ComponentStorage;

pub trait Bundle: 'static + Send + Sync {
    /// Returns an iterator over component IDs.
    fn component_ids() -> impl Iterator<Item = ComponentId>;

    /// Returns an iterator over component metadata.
    fn component_metas() -> impl Iterator<Item = ComponentMeta>;

    /// Writes data to the provided columns using direct index lookup.
    ///
    /// # Safety
    /// `columns` must be a slice of size `MAX_COMPONENTS`.
    /// `columns[id]` must be `Some` for every component `id` in this bundle.
    unsafe fn put<S: ComponentStorage>(self, columns: &mut [Option<S>]);
}

// Implement for individual Component
impl<T: Component> Bundle for T {
    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::once(T::get_id())
    }

    fn component_metas() -> impl Iterator<Item = ComponentMeta> {
        std::iter::once(T::meta())
    }

    unsafe fn put<S: ComponentStorage>(self, columns: &mut [Option<S>]) {
        // SAFETY: The caller guarantees that columns for all IDs returned by component_ids() exist.
        // T::get_id() is returned by component_ids().
        unsafe {
            let id = T::get_id();
            let col = columns.get_unchecked_mut(id.0).as_mut().unwrap_unchecked();
            col.submit(self);
        }
    }
}

// Implement for Unit
impl Bundle for () {
    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::empty()
    }

    fn component_metas() -> impl Iterator<Item = ComponentMeta> {
        std::iter::empty()
    }

    unsafe fn put<S: ComponentStorage>(self, _columns: &mut [Option<S>]) {}
}

// Implement for tuples
macro_rules! impl_bundle_tuple {
    ($($name:ident $num: tt),*) => {
        impl<$($name: Bundle),*> Bundle for ($($name,)*) {
            #[allow(non_snake_case)]
            fn component_ids() -> impl Iterator<Item = ComponentId> {
                std::iter::empty()
                $(
                    .chain($name::component_ids())
                )*
            }

            #[allow(non_snake_case)]
            fn component_metas() -> impl Iterator<Item = ComponentMeta> {
                std::iter::empty()
                $(
                    .chain($name::component_metas())
                )*
            }

            #[allow(non_snake_case)]
            unsafe fn put<Store: ComponentStorage>(self, columns: &mut [Option<Store>]) {
                unsafe {
                    $(
                        self.$num.put(columns);
                    )*
                }
            }
        }
    }
}

// Call the macro (available via #[macro_use] in lib.rs)
auto_impl_all_tuples!(impl_bundle_tuple);
S SV*cascade08VW WY*cascade08YZ Z_*cascade08_a ab*cascade08bv vw*cascade08wx x*cascade08	€ €*cascade08
‚ ‚„*cascade08
„… …†*cascade08
†‡ ‡ˆ*cascade08
ˆ‰ ‰‹*cascade08
‹Œ Œ*cascade08
   ¤*cascade08
¤¦ ¦¨*cascade08
¨© ©«*cascade08
«­ ­°*cascade08
°Ú Úá*cascade08
áâ âç*cascade08
çè èï*cascade08
ï *cascade08
‘ ‘’*cascade08
’“ “•*cascade08
•š šœ*cascade08
œ *cascade08
© ©«*cascade08
«¬ ¬­*cascade08
­® ®±*cascade08
±Ï ÏÖ*cascade08
Ö× ×Ü*cascade08
Üİ İä*cascade08
ä “*cascade08
“” ”–*cascade08
–— —*cascade08
Ÿ Ÿ *cascade08
 ¡ ¡©*cascade08
©« «°*cascade08
°² ²¸*cascade08
¸¹ ¹º*cascade08
º» »Ã*cascade08
Ãñ ñó*cascade08
óø øù*cascade08
ùú úû*cascade08
ûü üı*cascade08
ıÿ ÿ‰*cascade08
‰Š Š*cascade08
š š¢*cascade08
¢¤ ¤¥*cascade08
¥¬ ¬±*cascade08
±² ²¹*cascade08
¹º º¿*cascade08
¿À ÀÁ*cascade08
ÁÊ ÊË*cascade08
ËÍ ÍÎ*cascade08
ÎÏ ÏÑ*cascade08
Ñğ ğ…*cascade08
…› ›œ*cascade08
œ *cascade08
¡ ¡¤*cascade08
¤« «°*cascade08
°± ±·*cascade08
·¸ ¸º*cascade08
º» »¼*cascade08
¼¿ ¿Å*cascade08
ÅÇ ÇÊ*cascade08
ÊË ËÍ*cascade08
ÍÎ ÎÏ*cascade08
ÏÓ ÓØ*cascade08
ØÙ ÙŞ*cascade08
Şß ßà*cascade08
àâ âã*cascade08
ãä äå*cascade08
åë ëì*cascade08
ìï ïğ*cascade08
ğñ ñò*cascade08
òø øù*cascade08
ùú úû*cascade08
ûƒ ƒ‰*cascade08
‰Š ŠŒ*cascade08
Œ ‘*cascade08
‘’ ’“*cascade08
“– –˜*cascade08
˜š šœ*cascade08
œ Ÿ*cascade08
Ÿ   £*cascade08
£« «¬*cascade08
¬­ ­°*cascade08
°¸ ¸¹*cascade08
¹º º¼*cascade08
¼½ ½¿*cascade08
¿Â ÂÅ*cascade08
ÅÈ ÈÔ*cascade08
ÔÚ ÚÛ*cascade08
Ûä äê*cascade08
êë ëî*cascade08
îï ïğ*cascade08
ğñ ñó*cascade08
óş ş€*cascade08
€ „*cascade08
„† †ˆ*cascade08
ˆŠ Š‹*cascade08
‹Œ Œ*cascade08
’ ’•*cascade08
•– –˜*cascade08
˜™ ™œ*cascade08
œ£ £«*cascade08
«­ ­®*cascade08
®¯ ¯³*cascade08
³´ ´º*cascade08
º» »¼*cascade08
¼½ ½¿*cascade08
¿Á ÁÂ*cascade08
ÂÆ ÆÉ*cascade08
ÉÍ ÍÓ*cascade08
ÓÔ ÔÖ*cascade08
ÖØ ØÚ*cascade08
ÚÛ ÛÜ*cascade08
Üç çî*cascade08
îï ïõ*cascade08
õö öù*cascade08
ùú úû*cascade08
ûü üƒ*cascade08
ƒ„ „‰*cascade08
‰‹ ‹*cascade08
› ›*cascade08
 ¥*cascade08
¥¦ ¦©*cascade08
©ª ª­*cascade08
­¯ ¯³*cascade08
³´ ´µ*cascade08
µ¶ ¶º*cascade08
º¼ ¼À*cascade08
ÀÁ ÁÈ*cascade08
ÈÊ ÊÌ*cascade08
ÌÎ ÎÓ*cascade08
ÓÕ Õà*cascade08
àá áâ*cascade08
âê êí*cascade08
íî îğ*cascade08
ğñ ñ÷*cascade08
÷€	 €	‚	*cascade08
‚	ƒ	 ƒ		*cascade08
		 	‘	*cascade08
‘	’	 ’	“	*cascade08
“	–	 –	š	*cascade08
š	›	 ›		*cascade08
		 	Ÿ	*cascade08
Ÿ	§	 §	ª	*cascade08
ª	­	 ­	®	*cascade08
®	·	 ·	½	*cascade08
½	¾	 ¾	¿	*cascade08
¿	Ì	 Ì	Í	*cascade08
Í	Ï	 Ï	Ğ	*cascade08
Ğ	Ò	 Ò	Ö	*cascade08
Ö	î	 î	ñ	*cascade08
ñ	ò	 ò	õ	*cascade08
õ	ö	 ö	÷	*cascade08
÷	ø	 ø	–
*cascade08
–
—
 —
´
*cascade08
´
Á
 Á
È
*cascade08
È
É
 É
Ì
*cascade08
Ì
Í
 Í
Ò
*cascade08
Ò
Û
 Û
İ
*cascade08
İ
ã
 ã
ä
*cascade08
ä
å
 å
è
*cascade08
è
é
 é
ò
*cascade08
ò
ó
 ó
ö
*cascade08
ö
÷
 ÷
ø
*cascade08
ø
ú
 ú
û
*cascade08
û
ü
 ü
€*cascade08
€ ‡*cascade08
‡ˆ ˆ‹*cascade08
‹Œ Œ*cascade08
 ‘*cascade08
‘« «²*cascade08
²³ ³¸*cascade08
¸¹ ¹À*cascade08
ÀØ ØŞ*cascade08
Şà àä*cascade08
äå åé*cascade08
éî îğ*cascade08
ğõ õ÷*cascade08
÷ø øş*cascade08
şÿ ÿ€*cascade08
€ ‚*cascade08
‚ƒ ƒ„*cascade08
„… …‰*cascade08
‰Š Š*cascade08
 –*cascade08
–— —Ÿ*cascade08
Ÿ¬ ¬®*cascade08
®¯ ¯°*cascade08
°± ±²*cascade08
²» »¼*cascade08
¼¾ ¾Â*cascade08
ÂÃ ÃÄ*cascade08
ÄÇ ÇÉ*cascade08
ÉÊ ÊË*cascade08
ËÒ ÒÕ*cascade08
ÕÙ Ùß*cascade08
ßà àâ*cascade08
âã ãé*cascade08
éê ê÷*cascade08
÷ø øü*cascade08
üş ş‡*cascade08
‡ˆ ˆ‰*cascade08
‰Š Š•*cascade08
•– –œ*cascade08
œ ©*cascade08
©¬ ¬°*cascade08
°± ±²*cascade08
²³ ³¼*cascade08
¼½ ½À*cascade08
ÀÁ ÁÅ*cascade08
ÅÆ ÆÈ*cascade08
ÈÊ ÊÎ*cascade08
ÎÔ ÔÚ*cascade08
ÚÜ Üà*cascade08
àá áå*cascade08
åê êğ*cascade08
ğò òó*cascade08
óô ôø*cascade08
øù ùû*cascade08
ûü ü*cascade08
Š Š—*cascade08
—˜ ˜¢*cascade08
¢£ £©*cascade08
©ª ª­*cascade08
­® ®½*cascade08
½É ÉË*cascade08
ËÌ ÌÕ*cascade08
ÕÖ ÖØ*cascade08
ØÙ Ùİ*cascade08
İà àá*cascade08
áî îğ*cascade08
ğñ ñ€*cascade08
€ ƒ*cascade08
ƒ„ „ˆ*cascade08
ˆ‰ ‰–*cascade08
–— —˜*cascade08
˜™ ™¥*cascade08
¥¦ ¦§*cascade08
§¸ ¸Ê*cascade08
Êò òù*cascade08
ù€ €†*cascade08
†‡ ‡ˆ*cascade08
ˆŒ Œ*cascade08
 *cascade08
Ã ÃÅ*cascade08
ÅÆ ÆÍ*cascade08
ÍÎ ÎÓ*cascade08
ÓÔ ÔÕ*cascade08
Õü üƒ*cascade08
ƒ„ „‰*cascade08
‰Š Š‘*cascade08
‘² ²½*cascade08
½¾ ¾Õ*cascade08
Õì ìó*cascade08
óú ú„*cascade08
„ˆ ˆ‰*cascade08
‰‹ ‹Œ*cascade08
Œ½ ½¾*cascade08
¾À ÀÁ*cascade08
ÁÂ ÂÉ*cascade08
ÉÊ ÊÏ*cascade08
ÏĞ ĞÑ*cascade08
Ñí íï *cascade08ïó*cascade08óô *cascade08
ôõ õ‡*cascade08
‡œ œ*cascade08
¢ ¢¤ *cascade08¤¨*cascade08¨© *cascade08
©ö öû*cascade08ûü*cascade08
üı ış*cascade08
şÓ ÓÔ*cascade08
ÔÕ ÕÖ*cascade08
ÖÜ Üİ*cascade08
İá áâ*cascade08
âæ æç*cascade08
çè èé*cascade08
éê êë*cascade08ëì*cascade08ìğ*cascade08
ğñ ñó*cascade08
óô ôö*cascade08ö÷ *cascade08÷û*cascade08ûü*cascade08
ü€ €‚*cascade08
‚ƒ ƒŒ*cascade08
Œ *cascade08
° °¶*cascade08
¶» "(9ea8a383f98ac0eaf3b606baf12d22917ee922ae2Wfile:///home/jackc/Documents/code/rust/game_engine/crates/game_engine_ecs/src/bundle.rs:2file:///home/jackc/Documents/code/rust/game_engine