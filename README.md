# UniV
Blender Addon for UV Editor which aims to cover all sorts of operators for working with UV

[![Watch the video](https://i.ytimg.com/vi/-LEZl2Q9lhY/maxresdefault.jpg)](https://youtu.be/-LEZl2Q9lhY)

Buy:

 * [Gumroad](https://oxicid.gumroad.com/l/hblxa)

 * [Blender Market](https://blendermarket.com/products/univ?search_id=32308413)

Free Download:

 * [GitHub](https://github.com/Oxicid/UniV/releases)

 * [Blender Extension](https://extensions.blender.org/approval-queue/univ/)

[Discord](https://discord.gg/SAvEbGTkjR)

There is much more to UniV operators than you might think at first glance. 
Many operators are context-dependent, for example, on Sync state, selection mode (Verts, Edge, Face and Islands), as well as on pressed Ctrl, Shift, Alt (CSA) keys and combinations thereof. 

That is, before pressing the LMB button press CSA, then other modes of the operator are called. And these modifications are subject to a certain logic, which in __most cases__ works:
 + Ctrl - __To Cursor__ for transform or __Deselect__ for select
 + Alt - __Alternative__ operation that is fundamentally different from the default.
 + Shift - __Individual__, __Inplace__ for transform or __Extend__ for select

But you don't have to use the CSA keys, because a panel appears in the lower left corner where you can change the properties 

Also, the addon doesn't impose its hotkeys on you, but you can easily enable them in Settings->Extensions->UniV->Keymaps. But some operators due to their specificity can be called only through keymaps (QuickSnap, SplitUVToggle, SyncUVToggle).

## Operators
__Align Edge by Angle__ - align edges by angle in Island mode.

<img src="https://github.com/user-attachments/assets/41dbb904-f8c6-4efa-8338-40f629409c97" height="200">

__Quick Snap__ - allows you to transform an island or geometry elements by Vertex, Edge Center and Face Center. There are two methods of calling the operator via the keys:
 * __V__ - quick start, the nearest transformation element is selected immediately
 * __Alt+V__ - here you can safely select the necessary element and change the modes to __Island__ or __Element__ mode

<img src="https://github.com/user-attachments/assets/9ef6a4eb-f82b-4070-9794-75a1660d2340" height="200">

__Stack__ - overlap islands with similar geometry:

<img src="https://github.com/user-attachments/assets/dd521f1f-e6c3-4708-8b69-f42a3e2fce21" height="200">

__Weld__ - [W] connects selected vertices by common vertexes, if there is nothing to connect, it works as __Stitch__ 
<img src="https://github.com/user-attachments/assets/8da45e1f-36f8-4f6a-94a2-a9240c5773a2" height="200">

 * __Alt+Weld Button__ - connect by distance
  
   <img src="https://github.com/user-attachments/assets/f78a3492-c64b-435c-8878-6dda1b062999" height="200">

 __Stitch__ - [Shift+W] connects islands to each other, preserving the islands' proportions. When called via keymap with sync enabled, the target island becomes the closest:
* __Alt__ - finds all common island edges that have at least one __face__ selected, and connects through them.

<img src="https://github.com/user-attachments/assets/272e1186-a558-4c8a-9f70-505a38f30f29" height="200">

 __Stitch with padding__
 
<img src="https://github.com/user-attachments/assets/904bf34f-cacc-4ff9-abf1-6c2468de9d68" height="200">
<img src="https://github.com/user-attachments/assets/7f7f4e74-305d-410c-9ded-58b75de0bd12" height="200">

__Unwrap__ - differs from the built-in operator in that it unwrap inplace.

<img src="https://github.com/user-attachments/assets/55fd93cf-4ad1-46e6-a2be-df9f4fc861a0" height="200">

__Relax__ - combination of minimize stretch and unwrap borders.

<img src="https://github.com/user-attachments/assets/3a8c8d6c-87ac-4f24-8cdd-722aa4ab84e4" height="200">

__Cut__ - [C] - sets mark seams by border selection and at the same time expands the island

<img src="https://github.com/user-attachments/assets/214394e4-61ab-407a-b5f4-0ed6fd3a6e08" height="200">
<img src="https://github.com/user-attachments/assets/77391b12-9edf-413d-bca6-0e5e21eeaaa9" height="200">
<img src="https://github.com/user-attachments/assets/d5ad350d-fe91-49e7-a608-4faf8a8409fa" height="200">

__Quadrify__ - [E] align selected UV to rectangular distribution

<img src="https://github.com/user-attachments/assets/8bf353f6-8dc0-45bb-bd0f-ee269ea2165b" height="200">


__Straight__ - [Shift+E] straighten selected edge-chain and relax the rest of the UV Island

<img src="https://github.com/user-attachments/assets/c724dd15-e949-4529-aa93-5eb6e218fa59" height="200">

__Unwrap__ - [U] inplace unwrap 

<img src="https://github.com/user-attachments/assets/ad1bc69f-76eb-4d17-b29c-334f86bf36c4" height="200">

__Hide__ - [H] improved hide, with a more expected result
In the `default` Hide, islands that share common vertices are also affected.

<img src="https://github.com/user-attachments/assets/8fd1c911-b938-47bd-9fb8-e792a5193d21" height="200">
<img src="https://github.com/user-attachments/assets/cbe0c418-a0c8-4a79-a775-faf4b5f780ce" height="200">

The same issue occurs in `Non-Sync` mode when the mesh select mode is set to `Vertex & Edge`.
<img src="https://github.com/user-attachments/assets/0e178216-813b-43af-a1d8-6fdf16b8c041" height="200">
<img src="https://github.com/user-attachments/assets/0ce49db0-68df-4c2a-bc39-057551c80ec8" height="200">

__Pick Hide__

<img src="https://github.com/user-attachments/assets/461da666-72b9-43cb-8c26-98f351a7dc39" height="200">

__Sort__ - sorts islands, also aligns for more compactness.
 * __Axis__ - default is auto, which gives double priority to width, from the original boundary box
 * __Reverse__ - sorting by Increasing
 * __Orient__ - aligns the islands, making the sorting as compact as possible
 * __Overlapped__ - saves overlaps of islands
 * __Subgroups__ - creates subgroups of sorted islands by type

 __Distribute__ - similar to __Sort__, but the goal is to arrange the islands evenly according to the original position
 * __Spase__ - creates an even distance between the pivots of the islands
 * __Break__ - divides islands by angle, by Mark Seams, Mark Sharps, Material ID

__Copy to Layer__ - copy coordinates between uv-channels

<img src="https://github.com/user-attachments/assets/5574d195-5a61-45be-83c3-708379a8612c" height="200">

__Adjust & Normalize__ - sets a uniform texel

<img src="https://github.com/user-attachments/assets/bec51a5e-8e12-4f13-87b8-29831c91dd86" height="200">

__Flatten__ - convert 3d coords to 2D from uv map

<img src="https://github.com/user-attachments/assets/9353a702-cb90-401d-bf31-bf8661ac4d5d" height="200">

__Focus__ - [F] improved focusing, which frames the focussed elements.

<img src="https://github.com/user-attachments/assets/9d3e9e75-481a-424e-b610-9891ca9b9e6f" height="200">

__Crop__ - proportional tile filling

__Fill__ - it is similar in everything with __Crop__, but without preserving the proportions

__Orient__ - orients islands along the axes, maximally filling the area according to the boundbox. In a 3D viewport, it .

__Orient3D__ - orients islands according to world coordinates.

__Align__ - straightens, moves islands or vertices, places a 2D cursor on bound boxes, etc.

__Random__ - randomly transforms islands to break down repetitive patterns.

__Linked__ - [Ctrl+Shift+Mouse Scroll] - Select Linked

__Cursor__ - Select by Cursor

__Border__ - Select border edges

__Border by Angle__ - Select border edges by angle from 2D

__Inner__ - Select inner edges by mark sharps, angle

__Square__ - Select square, horizontal or vertical island

__SyncUVToggle__ - [~] toggle sync mode with element selection preserve

__SplitUVToggle__ - [Shift+T] open, close, toggle __UV__ or __3D View__  area 
