Our project is a simulation of a robot trying to find its way to a partner. The entire simulation takes place on a grass meadow, with its own “skybox”. A small portion of this meadow is editable and used in this simulation. The short summary for this is that the user edits the land, adding different objects, creating lakes and mountains, and then places the robot (whose name is Greg) anywhere they’d like, places Greg’s partner (whose name is Gretchen) anywhere they’d like, and then Greg will find his way around the obstacles in the terrain to Gretchen. 

![placed](https://user-images.githubusercontent.com/91294716/144538223-8d570eab-bbf2-4d5a-9c15-612663eca153.png)

For the interactive elements, the user will be able to choose places on the terrain to raise (by pressing ‘r’), lower (by pressing ‘l’) and occlude (by pressing ‘o’). When raising the terrain, there is a height limit that the user cannot pass, and you will notice that the hills and mountains create a shadow (advanced feature). When lowering the terrain, you will notice there is water beneath the grass/dirt, which has its own texture, reflections, refractions, and depth fog that you can inspect by zooming in. When occluding the grass, you can see the dirt underneath it. 

![water](https://user-images.githubusercontent.com/91294716/144538252-9792ed66-18bb-4617-b9f9-2c4f4c8a3ee3.gif)

The user can also place objects such as rocks (by pressing ‘1’) and trees (by pressing ‘2’), and it works the same way as raising and lowering the terrain, they pick a spot and choose which object to place. We have added textures (diffuse/specular/normal maps) to all these objects, and as one of our advanced features, they all have their own dynamic shadowing as well. These objects can be placed on the ground, in the water, on the hills, anywhere the user wants. 

![objects](https://user-images.githubusercontent.com/91294716/144538238-6ac5d4a5-9988-4152-8a42-1368888207f2.png)

Once the user is done editing the terrain, they can finally place Gretchen (by pressing ‘4’) and Greg (by pressing ‘3’). Once the user presses ‘0’, Greg will start moving towards Gretchen, while avoiding any rocks, trees, hills, or ponds along the way. Fun fact: Greg occludes the grass as he moves past it. We used A* for our pathfinding algorithm. 

![pathfinding](./assets/pathfinding-min.gif)

Attributions:

OBJ files and misc textures: https://opengameart.org/

A* code: https://briangrinstead.com/blog/astar-search-algorithm-in-javascript-updated/

Water textures: https://catlikecoding.com/unity/tutorials/flow/texture-distortion/

Water Shader inspiration:

https://halfpastyellow.com/blog/2020/10/01/Yet-Another-Stylised-Water-Shader.html

https://www.youtube.com/watch?v=4FIDBeF_4SI&t=828s

Grass Shader Inspiration: https://xbdev.net/directx3dx/specialX/Fur/


Misc:

Current uploaded version of the project has lowered visual fidelity for the sake of performance. For full fidelity rollback to commit id: 74ce3e333d8aeac4164d6b283e57aa0d3500ea73. This is commit "optimizations" on dec 1st.
