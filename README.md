# UniV
Blender Addon for UV Editor which aims to cover all sorts of operators for working with UV

[![Watch the video](https://i.ytimg.com/vi/KxYE8IHpVbE/maxresdefault.jpg)](https://youtu.be/KxYE8IHpVbE)

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
__Quick Snap__ - allows you to transform an island or geometry elements by Vertex, Edge Center and Face Center. There are two methods of calling the operator via the keys:
 * __V__ - quick start, the nearest transformation element is selected immediately
 * __Alt+V__ - here you can safely select the necessary element and change the modes to __Island__ or __Element__ mode

<img src="https://github.com/user-attachments/assets/9ef6a4eb-f82b-4070-9794-75a1660d2340" width="350">

__Stack__ - overlap islands with similar geometry:

<img src="https://github.com/user-attachments/assets/dd521f1f-e6c3-4708-8b69-f42a3e2fce21" width="350">

__Weld__ - [W] connects selected vertices by common vertexes, if there is nothing to connect, it works as __Stitch__ 
<img src="https://github.com/user-attachments/assets/8da45e1f-36f8-4f6a-94a2-a9240c5773a2" width="350">

 * __Alt+Weld Button__ - connect by distance
  
   <img src="https://github.com/user-attachments/assets/f78a3492-c64b-435c-8878-6dda1b062999" width="350">

 __Stitch__ - [Shift+W] connects islands to each other, preserving the islands' proportions. When called via keymap with sync enabled, the target island becomes the closest:
* __Alt__ - finds all common island edges that have at least one __face__ selected, and connects through them.

<img src="https://github.com/user-attachments/assets/272e1186-a558-4c8a-9f70-505a38f30f29" width="350">

__Unwrap__ - differs from the built-in operator in that it unwrap inplace.

<img src="https://github.com/user-attachments/assets/55fd93cf-4ad1-46e6-a2be-df9f4fc861a0" width="350">

__Relax__ - combination of minimize stretch and unwrap borders.

__Cut__ - [C] - sets mark seams by border selection and at the same time expands the island

__Quadrify__ - [E] align selected UV to rectangular distribution

<img src="https://github.com/user-attachments/assets/8bf353f6-8dc0-45bb-bd0f-ee269ea2165b" width="350">


__Straight__ - [Shift+E] straighten selected edge-chain and relax the rest of the UV Island

<img src="https://github.com/user-attachments/assets/c724dd15-e949-4529-aa93-5eb6e218fa59" width="350">


__Sort__ - sorts islands, also aligns for more compactness.
 * __Axis__ - default is auto, which gives double priority to width, from the original boundary box
 * __Reverse__ - sorting by Increasing
 * __Orient__ - aligns the islands, making the sorting as compact as possible
 * __Overlapped__ - saves overlaps of islands
 * __Subgroups__ - creates subgroups of sorted islands by type

 __Distribute__ - similar to __Sort__, but the goal is to arrange the islands evenly according to the original position
 * __Spase__ - creates an even distance between the pivots of the islands
 * __Break__ - divides islands by angle, by Mark Seams, Mark Sharps, Material ID

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
