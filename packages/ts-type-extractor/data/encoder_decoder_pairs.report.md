# Per-Rule Breakdown

| idx | Rule | muted | total | pos | neg | train | val | recov | full | example |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| 11b | `html_specific_element→html_element` | no | 5343 | 4180 | 1163 | 4554 | 789 | 3% | 3% | `HTMLInputElement -> HTMLElement` |
| 13 | `indexed_access_type→unknown` | no | 4824 | 3923 | 901 | 4073 | 751 | 15% | 2% | `ButtonProps['variant'] -> unknown` |
| 14 | `utility_type→unknown` | no | 4611 | 3719 | 892 | 3897 | 714 | 12% | 9% | `Partial<User> -> unknown` |
| 10 | `string_literal_union→string` | no | 3825 | 2976 | 849 | 3239 | 586 | 4% | 3% | `'primary' | 'secondary' | 'danger' -> string` |
| 2 | `react_event→synthetic` | no | 2157 | 2089 | 68 | 1810 | 347 | 1% | 0% | `React.MouseEvent<HTMLButtonElement> -> React.SyntheticEvent` |
| 11e | `record_string_value→unknown` | no | 1615 | 1338 | 277 | 1379 | 236 | 10% | 10% | `Record<string, string> -> Record<string, unknown>` |
| 11c | `html_specific_element_nullable→html_element_nullable` | no | 699 | 539 | 160 | 592 | 107 | 8% | 8% | `HTMLInputElement | null -> HTMLElement | null` |
| 15 | `promise→unknown` | no | 627 | 608 | 19 | 529 | 98 | 15% | 13% | `Promise<User> -> Promise<unknown>` |
| 6 | `react_refobject→unknown` | no | 606 | 606 | 0 | 525 | 81 | 27% | 18% | `React.RefObject<HTMLDivElement> -> React.RefObject<unknown>` |
| 11g | `set→unknown` | no | 398 | 395 | 3 | 335 | 63 | 37% | 35% | `Set<string> -> Set<unknown>` |
| 11f | `map→unknown` | no | 376 | 375 | 1 | 315 | 61 | 34% | 26% | `Map<string, number> -> Map<unknown, unknown>` |
| 1 | `react_event_handler→generic` | no | 235 | 235 | 0 | 200 | 35 | 32% | 25% | `React.MouseEventHandler<HTMLButtonElement> -> React.EventHandler<React.SyntheticEvent>` |
| 12 | `conditional_type→unknown` | **yes** | 215 | 172 | 43 | 185 | 30 | 25% | 6% | `T extends string ? 'text' : 'other' -> unknown` |
| 7 | `react_mutable_refobject→unknown` | no | 181 | 180 | 1 | 151 | 30 | 28% | 21% | `React.MutableRefObject<boolean> -> React.MutableRefObject<unknown>` |
| 11 | `template_literal_type→string` | no | 176 | 137 | 39 | 146 | 30 | 24% | 20% | `` `--${string}` -> string `` |
| 8 | `react_dispatch_setstateaction→unknown` | no | 115 | 115 | 0 | 102 | 13 | 25% | 19% | `React.Dispatch<React.SetStateAction<string>> -> React.Dispatch<React.SetStateAction<unknown>>` |
| 1b | `react_specific_event_handler_alias→generic` | no | 75 | 75 | 0 | 60 | 15 | 15% | 15% | `MouseEventHandler<HTMLButtonElement> -> React.EventHandler<React.SyntheticEvent>` |
| 11d | `custom_event→unknown` | no | 56 | 56 | 0 | 49 | 7 | 5% | 4% | `CustomEvent<{ action: string; payload: unknown }> -> CustomEvent<unknown>` |
| 16 | `readonly_array→unknown` | **yes** | 52 | 51 | 1 | 47 | 5 | 19% | 12% | `ReadonlyArray<string> -> ReadonlyArray<unknown>` |
| 14e | `dom_css_style_declaration→unknown` | **yes** | 47 | 35 | 12 | 40 | 7 | 3% | 3% | `CSSStyleDeclaration -> unknown` |
| 14d | `dom_shadow_root_init→unknown` | **yes** | 25 | 20 | 5 | 22 | 3 | 0% | 0% | `ShadowRootInit -> unknown` |
| 14c | `dom_intersection_observer_init→unknown` | **yes** | 15 | 12 | 3 | 13 | 2 | 0% | 0% | `IntersectionObserverInit -> unknown` |
| 19 | `astro_collection_entry→any` | **yes** | 14 | 14 | 0 | 12 | 2 | 0% | 0% | `CollectionEntry<'blog'> -> CollectionEntry<any>` |
| 14b | `dom_mutation_observer_init→unknown` | **yes** | 13 | 10 | 3 | 12 | 1 | 0% | 0% | `MutationObserverInit -> unknown` |
| 11h | `dom_add_event_listener_options→event_listener_options` | **yes** | 12 | 11 | 1 | 10 | 2 | 9% | 9% | `AddEventListenerOptions -> EventListenerOptions` |
| 4 | `react_component_props_without_ref→any` | **yes** | 12 | 12 | 0 | 11 | 1 | 0% | 0% | `React.ComponentPropsWithoutRef<'input'> -> React.ComponentPropsWithoutRef<any>` |
| 9 | `jsx_intrinsic_keyof→string` | **yes** | 11 | 8 | 3 | 7 | 4 | 0% | 0% | `keyof JSX.IntrinsicElements -> string` |
| 3 | `react_component_props_with_ref→any` | **yes** | 7 | 7 | 0 | 7 | 0 | 7% | 0% | `React.ComponentPropsWithRef<'button'> -> React.ComponentPropsWithRef<any>` |
| 18g | `astro_api_route→unknown` | **yes** | 6 | 4 | 2 | 4 | 2 | 0% | 0% | `APIRoute -> unknown` |
| 5 | `react_element_ref→any` | **yes** | 3 | 3 | 0 | 3 | 0 | 0% | 0% | `React.ElementRef<typeof Button> -> React.ElementRef<any>` |
| 18c | `tanstack_infinite_data→unknown` | **yes** | 3 | 3 | 0 | 3 | 0 | 0% | 0% | `InfiniteData<Post[], number> -> InfiniteData<unknown, unknown>` |
| 18e | `astro_infer_get_static_props_type→unknown` | **yes** | 2 | 2 | 0 | 2 | 0 | 0% | 0% | `InferGetStaticPropsType<typeof getStaticProps> -> InferGetStaticPropsType<unknown>` |
| 14f | `dom_element_internals_intersection→unknown` | **yes** | 2 | 1 | 1 | 2 | 0 | 0% | 0% | `ElementInternals & { form: HTMLFormElement } -> unknown` |
