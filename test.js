import {defs, tiny} from './examples/common.js';
import {
    Skybox_Shader,
    PlainShader,
    Grass_Shader_Shadow,
    Grass_Shader_Background,
    Phong_Water_Shader,
    Grass_Shader_Shadow_Textured,
    Shadow_Textured_Phong, Shadow_Textured_Phong_Maps, Grass_Shader_Background_Shadow
} from './shaders.js';
import {Triangle_Strip_Plane, Dynamic_Texture, Custom_Movement_Controls, Buffered_Texture, Scene_Object, Maze_Solver, Shape_From_File} from './utils.js';
const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Matrix, Mat4, Light, Shape, Material, Scene, Shader, Graphics_Card_Object, Texture
} = tiny;

export class Team_Project extends Scene {
    make_control_panel(context) {
        this.key_triggered_button("raise terrain", ["r"], () => {
            this.isRaising = true;
            this.isLowering = false;
            this.isOccluding = false;
            this.placeRock = false;
            this.placeTree = false;
            this.solveMaze = false;
        });
        this.key_triggered_button("lower terrain", ["l"], () => {
            this.isRaising = false;
            this.isLowering = true;
            this.isOccluding = false;
            this.placeRock = false;
            this.placeTree = false;
            this.solveMaze = false;
        });
        this.key_triggered_button("occlude grass", ["o"], () => {
            this.isRaising = false;
            this.isLowering = false;
            this.isOccluding = true;
            this.placeRock = false;
            this.placeTree = false;
            this.solveMaze = false;
        });
        this.key_triggered_button("place rock", ["1"], () => {
            this.isRaising = false;
            this.isLowering = false;
            this.isOccluding = false;
            this.placeRock = true;
            this.placeTree = false;
            this.solveMaze = false;
        });
        this.key_triggered_button("place tree", ["2"], () => {
            this.isRaising = false;
            this.isLowering = false;
            this.isOccluding = false;
            this.placeRock = false;
            this.placeTree = true;
            this.solveMaze = false;
        });
        this.key_triggered_button("Solve Maze", ["3"], () => {
            this.isRaising = false;
            this.isLowering = false;
            this.isOccluding = false;
            this.placeRock = false;
            this.placeTree = false;
            this.solveMaze = true;
        });
    }

    texture_buffer_init(gl) {
        this.lightDepthTextureGPU = gl.createTexture();
        this.lightDepthTexture = new Buffered_Texture(this.lightDepthTextureGPU);
        this.cameraDepthTextureGPU = gl.createTexture();
        this.cameraDepthTexture = new Buffered_Texture(this.cameraDepthTextureGPU);
        this.cameraColorTextureGPU = gl.createTexture();
        this.cameraColorTexture = new Buffered_Texture()

        //light

        gl.bindTexture(gl.TEXTURE_2D, this.lightDepthTextureGPU);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT, this.lightDepthTextureSize, this.lightDepthTextureSize, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        this.lightDepthFramebuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.lightDepthFramebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.lightDepthTextureGPU, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        //camera

        gl.bindTexture(gl.TEXTURE_2D, this.cameraDepthTextureGPU);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT, 1920, 1080, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, this.cameraColorTextureGPU);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1920, 1080, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        this.cameraDepthFramebuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.cameraDepthFramebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.cameraDepthTextureGPU, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.cameraColorTextureGPU, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        // create a color texture of the same size as the depth texture
        // see article why this is needed_
        // this.unusedTexture = gl.createTexture();
        // gl.bindTexture(gl.TEXTURE_2D, this.unusedTexture);
        // gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.lightDepthTextureSize, this.lightDepthTextureSize,
        //     0, gl.RGBA, gl.UNSIGNED_BYTE, null,);
        // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        // // attach it to the framebuffer
        // gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.unusedTexture, 0);
        // gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    //helper function to get the location of the closest vertex on our plane to where the mouse is pointing, can care about or disregard yAxis position of the plane
    getClosestLocOnPlane(plane, context, program_state, yAxis = true) {
        //get the size of our canvas so we can know how far the mouse is from the center by normalizing as a percent from -1-1
        let rect = context.canvas.getBoundingClientRect();
        let mousePosPercent = Vector.create(2 * context.scratchpad.controls.mouse.from_center[0] / (rect.right - rect.left),
            -2 * context.scratchpad.controls.mouse.from_center[1] / (rect.bottom + rect.top));

        //to turn this percentage into a usable value in our coordinate space we need to transform it from clip space to world space
        //we can do this by multiplying and inverting our camera and projection matrices
        let transformMatrix = Mat4.inverse(program_state.projection_transform.times(program_state.camera_inverse));
        //we now get two point in clip space, on on the near part of the cube, and one on the far part of the cube
        let mousePointNear = Vector.create(mousePosPercent[0], mousePosPercent[1], -1, 1);
        let worldSpaceNear = transformMatrix.times(mousePointNear);
        //this is the world space coordinates of the near point. We divide by the last homogenous value because of the perspective transform
        worldSpaceNear = Vector.create(worldSpaceNear[0] / worldSpaceNear[3], worldSpaceNear[1] / worldSpaceNear[3], worldSpaceNear[2] / worldSpaceNear[3], 1);

        let mousePointFar = Vector.create(mousePosPercent[0], mousePosPercent[1], 1, 1);
        let worldSpaceFar = transformMatrix.times(mousePointFar);
        //this is the world space coordinates of the far point. We divide by the last homogenous value because of the perspective transform
        worldSpaceFar = Vector.create(worldSpaceFar[0] / worldSpaceFar[3], worldSpaceFar[1] / worldSpaceFar[3], worldSpaceFar[2] / worldSpaceFar[3], 1);

        //this calls the intersection function of the plane shape to find which vertex is nearest to the mouse click. the function takes
        //our mouse's location as the origin, and a vector direction to the world space location of the far coordinate, also a bool to see if to count yAxis position or not
        return plane.shape.closestVertexToRay(worldSpaceNear, worldSpaceFar.minus(worldSpaceNear).normalized(), yAxis);
    }

    //function that draws on a given texture at a given location with a given brush radius in pixels. need to pass in the world
    //mapping of the textures size. For example, if we want the texture to cover from -10 to 10 on the x and z axies, we pass in a width and
    //height of 20. This code assumes the texture is centered at the origin. This can be extended to an arbitrary center if we want
    //currently only draws on the red channel
    drawnOnTexture(texture, length, width, location, brushRadius) {
        //map from location on the plane, to location on the texture. Same way we do viewport and window transforms
        let textureLocPercent = Vector.create((location[0]-1) / (width / 2), -(location[2]-1) / (length / 2));
        let textureLoc = Vector.create(Math.ceil(textureLocPercent[0] * (texture.width / 2)) + (texture.width / 2), Math.ceil(textureLocPercent[1] * (texture.length / 2)) + (texture.length / 2));

        //use a line by line circle algorithm to draw a circle at the center with a radius of brushradius.
        //draws the circle multiple times with radius from 0 to brushradius so that the center of the circle has a
        //higher value than the lower parts of the circle
        let strength = 20;
        for (let i = 0; i < brushRadius; i++) {
            for (let dy = 0; dy < i; dy++) {
                let dx = 0;
                let sq = (i * i) - (dy * dy);
                while ((dx * dx) < sq) {
                    texture.data[((dx + textureLoc[0]) * 4) + ((dy + textureLoc[1]) * 4 * texture.width)] += strength;
                    texture.data[((dx + textureLoc[0]) * 4) + ((-dy + textureLoc[1] - 1) * 4 * texture.width)] += strength;
                    texture.data[((-dx + textureLoc[0] - 1) * 4) + ((dy + textureLoc[1]) * 4 * texture.width)] += strength;
                    texture.data[((-dx + textureLoc[0] - 1) * 4) + ((-dy + textureLoc[1] - 1) * 4 * texture.width)] += strength;
                    dx++;
                }
            }
        }
    }

    //lowers a given plane's vertices in a circle given by brushradius around a point. uses the same logic as the draw on texture function
    lowerPlane(plane, location, brushRadius) {
        let planeLocPercent = Vector.create((location[0]-1) / (plane.shape.length / 2), (location[2]-1) / (plane.shape.width / 2));
        planeLocPercent[0] = Math.max(Math.min(0.75, planeLocPercent[0]), -0.75);
        planeLocPercent[1] = Math.max(Math.min(0.75, planeLocPercent[1]), -0.75);
        let planeLoc = Vector.create(Math.ceil(planeLocPercent[0] * (plane.shape.length * plane.shape.density / 2)) + (plane.shape.length * plane.shape.density / 2),
            Math.ceil(planeLocPercent[1] * (plane.shape.width * plane.shape.density / 2)) + (plane.shape.width * plane.shape.density / 2));

        //attenuate strength based on brush radius so that large brushes don't raise terrain much faster than small brushes
        let strength = 0.1 / Math.max((brushRadius - 6), 1);
        for (let i = 7; i < brushRadius; i+=2) {
            for (let dy = 0; dy < i; dy++) {
                let dx = 0;
                let sq = (i * i) - (dy * dy);
                while ((dx * dx) < sq) {
                    plane.shape.removeVertexHeight(dx + planeLoc[0], dy + planeLoc[1], strength);
                    plane.shape.removeVertexHeight(dx + planeLoc[0], -dy + planeLoc[1] - 1, strength);
                    plane.shape.removeVertexHeight(-dx + planeLoc[0] - 1, dy + planeLoc[1], strength);
                    plane.shape.removeVertexHeight(-dx + planeLoc[0] - 1, -dy + planeLoc[1] - 1, strength);
                    dx++;
                }
            }
        }
    }

    //exact same implementation as lowerPlane except it raises the plane
    raisePlane(plane, location, brushRadius) {
        let planeLocPercent = Vector.create((location[0]-1) / (plane.shape.length / 2), (location[2]-1) / (plane.shape.width / 2));
        planeLocPercent[0] = Math.max(Math.min(0.75, planeLocPercent[0]), -0.75);
        planeLocPercent[1] = Math.max(Math.min(0.75, planeLocPercent[1]), -0.75);
        let planeLoc = Vector.create(Math.ceil(planeLocPercent[0] * (plane.shape.length * plane.shape.density / 2)) + (plane.shape.length * plane.shape.density / 2),
            Math.ceil(planeLocPercent[1] * (plane.shape.width * plane.shape.density / 2)) + (plane.shape.width * plane.shape.density / 2));
        let strength = 0.05 / Math.max((brushRadius - 6), 1);
        for (let i = 5; i < brushRadius; i++) {
            for (let dy = 0; dy < i; dy++) {
                let dx = 0;
                let sq = (i * i) - (dy * dy);
                while ((dx * dx) < sq) {
                    plane.shape.addVertexHeight(dx + planeLoc[0], dy + planeLoc[1], strength);
                    plane.shape.addVertexHeight(dx + planeLoc[0], -dy + planeLoc[1] - 1, strength);
                    plane.shape.addVertexHeight(-dx + planeLoc[0] - 1, dy + planeLoc[1], strength);
                    plane.shape.addVertexHeight(-dx + planeLoc[0] - 1, -dy + planeLoc[1] - 1, strength);
                    dx++;
                }
            }
        }
    }

    setFrameBufferForShadows(context, program_state) {
        context.context.bindFramebuffer(context.context.FRAMEBUFFER, this.lightDepthFramebuffer);
        context.context.viewport(0, 0, this.lightDepthTextureSize, this.lightDepthTextureSize);
        context.context.clear(context.context.COLOR_BUFFER_BIT | context.context.DEPTH_BUFFER_BIT);
        program_state.projection_transform = this.light_proj_mat;
        program_state.camera_inverse = this.light_view_mat;
    }

    setFrameBufferForDepthPass(context, program_state, projTransformStorage, cameraStorage){
        context.context.bindFramebuffer(context.context.FRAMEBUFFER, this.cameraDepthFramebuffer);
        context.context.viewport(0, 0, 1920, 1080);
        context.context.clear(context.context.COLOR_BUFFER_BIT | context.context.DEPTH_BUFFER_BIT);
        program_state.projection_transform = projTransformStorage;
        program_state.camera_inverse = cameraStorage;
    }


    resetFrameBuffer(context, program_state) {
        context.context.bindFramebuffer(context.context.FRAMEBUFFER, null);
        context.context.clear(context.context.COLOR_BUFFER_BIT | context.context.DEPTH_BUFFER_BIT);
        context.context.viewport(0, 0, context.context.canvas.width, context.context.canvas.height);
    }

    constructor() {
        super();

        this.Maze_Solver = new Maze_Solver();

        //creates a blank custom texture for the grass occlusion
        this.grassOcclusionTexture = new Dynamic_Texture(256, 256);
        this.rockDiffuseTexture = new Texture("assets/stone/stone_albedo.png");
        this.rockSpecularTexture = new Texture("assets/stone/stone_specular.png");
        this.rockNormalTexture = new Texture("assets/stone/stone_normal.png");
        this.treeDiffuseTexture = new Texture("assets/palm_tree/diffus.png");
        this.treeSpecularTexture = new Texture("assets/palm_tree/specular.png");
        this.treeNormalTexture = new Texture("assets/palm_tree/normal.png");

        this.shapes = {
            'axis' : new defs.Axis_Arrows(),
            "tree": new Shape_From_File("assets/palm_tree/palm.obj"),
            "rock": new Shape_From_File("assets/stone/stone_1.obj"),
        };

        this.shapesArray = [];

        //member variables to track our interaction state
        this.isRaising = true;
        this.isLowering = false;
        this.isOccluding = false;
        this.placeRock = false;
        this.placeTree = false;
        this.solveMaze = false;

        //grass vars
        this.grass_color = hex_color("#2d8f06");
        this.ground_color = hex_color("#556208");

        //variables to deal with light and shadows
        this.lightDepthTextureSize = 4096;
        this.light_position = vec4(23, 20, -23, 0);
        this.light_color = color(5,5,5,0);
        this.light_view_target = vec4(0, 0, 0, 1);
        this.light_field_of_view = 130 * Math.PI / 180;
        this.light_view_mat = Mat4.look_at(
            vec3(this.light_position[0], this.light_position[1], this.light_position[2]),
            vec3(this.light_view_target[0], this.light_view_target[1], this.light_view_target[2]),
            vec3(0, 1, 0),
        );
        this.light_proj_mat = Mat4.perspective(this.light_field_of_view, 1, 0.5, 500);

        this.materials = {
            plastic: new Material(new defs.Phong_Shader(), {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            plastic_shadows: new Material(new Shadow_Textured_Phong(), {ambient: .4, diffusivity: .6, specularity: 0.2, smoothness: 5, color_texture: new Texture("assets/noise/perlin2.png"),
                light_depth_texture: null, lightDepthTextureSize: this.lightDepthTextureSize, draw_shadow: true, light_view_mat: this.light_view_mat, light_proj_mat: this.light_proj_mat}),
            plain: new Material(new PlainShader()),

            rock: new Material(new Shadow_Textured_Phong_Maps(), {color_texture: this.rockDiffuseTexture, specular_texture: this.rockSpecularTexture, normal_texture: this.rockNormalTexture, ambient: 0.4, specularity: 0.3, diffusivity: 0.3, smoothness: 1,
                light_depth_texture: null, lightDepthTextureSize: this.lightDepthTextureSize, draw_shadow: true, light_view_mat: this.light_view_mat, light_proj_mat: this.light_proj_mat}),
            tree: new Material(new Shadow_Textured_Phong_Maps(), {color_texture: this.treeDiffuseTexture, specular_texture: this.treeSpecularTexture, normal_texture: this.treeNormalTexture, ambient: 0.4, specularity: 0.6, diffusivity: 0.45, smoothness: 5,
                light_depth_texture: null, lightDepthTextureSize: this.lightDepthTextureSize, draw_shadow: true, light_view_mat: this.light_view_mat, light_proj_mat: this.light_proj_mat}),
        };

        //create the background grass plane. low density since we aren't deforming it
        this.background_grass_plane = new Scene_Object(new Triangle_Strip_Plane(20, 20, Vector3.create(0,0,0), 2),
            Mat4.scale(5,1,5), new Material(new Grass_Shader_Background_Shadow(0), {grass_color: this.grass_color, ground_color: this.ground_color,
                ambient: 0.2, diffusivity: 0.3, specularity: 0.032, smoothness: 100, fake_shadow_layer: false,
                light_depth_texture: null, lightDepthTextureSize: this.lightDepthTextureSize, draw_shadow: true, light_view_mat: this.light_view_mat, light_proj_mat: this.light_proj_mat}), "TRIANGLE_STRIP");

        this.water_plane = new Scene_Object(new Triangle_Strip_Plane(5,5, Vector3.create(0,0,0), 5), Mat4.translation(-10,-0.7,-10).times(Mat4.scale(10,1,10)),
            new Material(new Phong_Water_Shader(), {shallow_color: hex_color("#0bc9da"), deep_color: hex_color("#213ebd"), ambient: 0.0, diffusivity: 0.0, specularity: 0.7, smoothness: 100,
            depth_texture: null, bg_color_texture: null}), "TRIANGLE_STRIP");

        //the main grass plane has a higher density since we want the deformation to look smooth
        this.grass_plane = new Scene_Object(new Triangle_Strip_Plane(26, 26, Vector3.create(0,0,0), 7),
            Mat4.translation(0,0,0), new Material(new Grass_Shader_Shadow(0), {grass_color: this.grass_color, ground_color: this.ground_color,
                texture: this.grassOcclusionTexture, ambient: 0.2, diffusivity: 0.3, specularity: 0.032, smoothness: 100, fake_shadow_layer: false,
                light_depth_texture: null, lightDepthTextureSize: this.lightDepthTextureSize, draw_shadow: true, light_view_mat: this.light_view_mat, light_proj_mat: this.light_proj_mat}), "TRIANGLE_STRIP");

        // this.noiseTexture1 = new Texture('assets/Blue.png', "LINEAR");
        // this.noiseTexture2 = new Texture('assets/Perlin2.png', "LINEAR");
        //
        // this.grass_plane = new Scene_Object(new Triangle_Strip_Plane(26, 26, Vector3.create(0,0,0), 7),
        //     Mat4.translation(0,0,0), new Material(new Grass_Shader_Shadow_Textured(0), {grass_color: hex_color("#38af18"), ground_color: hex_color("#544101"),
        //         texture: this.grassOcclusionTexture, ambient: 0.2, diffusivity: 0.3, specularity: 0.032, smoothness: 100, fake_shadow_layer: false,
        //         light_depth_texture: null, lightDepthTextureSize: this.lightDepthTextureSize, draw_shadow: true, light_view_mat: this.light_view_mat, light_proj_mat: this.light_proj_mat,
        //         noiseTexture1: this.noiseTexture1, noiseTexture2: this.noiseTexture2}), "TRIANGLE_STRIP");

        //the skybox is just a sphere with the shader that makes the color look vaguely like sky above. We put everything inside this sphere
        this.skybox = new Scene_Object(new defs.Subdivision_Sphere(4), Mat4.scale(80, 80,80),
            new Material(new Skybox_Shader(), {top_color: hex_color("#268b9a"), mid_color: hex_color("#d1eaf6"), bottom_color: hex_color("#3d8f2b")}));

        this.pathArr = [];
        this.obstacleArr = [];
        for (let i = 0; i < 26*7; i++){
            for(let j = 0; j < 26*7; j++){
                if (j === 0){
                    this.obstacleArr.push([]);
                }
                this.obstacleArr[i].push(1);
            }
        }
    }

    render_scene(context, program_state, depthPass, drawGrassDepth = false, drawBackgroundGrass = true) {
        const t = program_state.animation_time;

        if(depthPass === false) {
            this.skybox.drawObject(context, program_state);
            if (this.materials.plastic_shadows.light_depth_texture == null) {
                this.materials.plastic_shadows.light_depth_texture = this.lightDepthTexture;
            }
            this.shapes.axis.draw(context, program_state, Mat4.identity(), this.materials.plastic_shadows);



            //with the way the grass shader works, you need to draw it a number of times, each time specifying which level of the grass you want to draw
            //for the background 8-12 layers look good
            if (this.background_grass_plane.material.light_depth_texture == null) {
                this.background_grass_plane.material.light_depth_texture = this.lightDepthTexture;
            }
            this.background_grass_plane.material.draw_shadow = true;
            for (let i = 0; i < 18; i+= 2) {
                this.background_grass_plane.material.shader.layer = i;
                this.background_grass_plane.drawObject(context, program_state);
                // this.background_grass_plane.material.fake_shadow_layer = true;
                // this.background_grass_plane.drawObject(context, program_state);
                // this.background_grass_plane.material.fake_shadow_layer = false;
            }

            //16 layers looks good for the main grass portion. can increase or decrease later if we want
            if(this.grass_plane.material.light_depth_texture == null) {
                this.grass_plane.material.light_depth_texture = this.lightDepthTexture;
            }
            this.grass_plane.material.draw_shadow = true;
            for (let i = 0; i < 18; i+= 2) {
                this.grass_plane.material.shader.layer = i;
                this.grass_plane.drawObject(context, program_state);
                // this.grass_plane.material.fake_shadow_layer = true;
                // this.grass_plane.drawObject(context, program_state);
                // this.grass_plane.material.fake_shadow_layer = false;
            }
            if (this.materials.tree.light_depth_texture == null) {
                this.materials.tree.light_depth_texture = this.lightDepthTexture;
                this.materials.rock.light_depth_texture = this.lightDepthTexture;
            }
            for (let i = 0; i < this.shapesArray.length; i++)
            {
                this.shapesArray[i].drawObject(context, program_state);
            }
            if (this.water_plane.material.bg_color_texture == null) {
                this.water_plane.material.bg_color_texture = this.cameraColorTexture;
                this.water_plane.material.depth_texture = this.cameraDepthTexture;
            }
            this.water_plane.drawObject(context, program_state);

        }
        else{
            this.shapes.axis.draw(context, program_state, Mat4.identity(), this.materials.plain);
            if (drawGrassDepth){
                if (drawBackgroundGrass){
                    this.background_grass_plane.material.draw_shadow = false;
                    for (let i = 0; i < 12; i+= 3) {
                        this.background_grass_plane.material.shader.layer = i;
                        this.background_grass_plane.drawObject(context, program_state);
                    }
                }

                this.grass_plane.material.draw_shadow = false;
                for (let i = 0; i < 14; i+= 2) {
                    this.grass_plane.material.shader.layer = i;
                    this.grass_plane.drawObject(context, program_state);
                }
            }
            else {
                this.grass_plane.drawOverrideMaterial(context, program_state, this.materials.plain);
                if (drawBackgroundGrass){
                    this.background_grass_plane.drawOverrideMaterial(context, program_state, this.materials.plain);
                }
            }


            for (let i = 0; i < this.shapesArray.length; i++)
            {
                this.shapesArray[i].drawOverrideMaterial(context, program_state, this.materials.plain);
            }
        }
    }

    display(context, program_state) {
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new Custom_Movement_Controls());
            program_state.set_camera(Mat4.look_at(vec3(9, 15, 22), vec3(1, 0, 0), vec3(0, 1, 0)));
            program_state.projection_transform = Mat4.perspective(Math.PI / 4, context.width / context.height, 1, 500);

            const ext = context.context.getExtension('WEBGL_depth_texture');
            if (!ext) {
                return alert('need WEBGL_depth_texture');  // eslint-disable-line
            }
            this.texture_buffer_init(context.context);
        }

        program_state.lights = [new Light(this.light_position, this.light_color, 10000)];

        //if the mouse is held down and we are trying to paint on the canvas
        if (context.scratchpad.controls.mouse.isPainting) {
            //if we are raising terrain, find out location on the plane without considering y axis and then edit the shape and copy the new shape to the graphics card
            if (this.isRaising === true) {
                let dest = this.getClosestLocOnPlane(this.grass_plane, context, program_state, false);
                this.raisePlane(this.grass_plane, dest, 20);
                this.grass_plane.shape.copy_onto_graphics_card(context.context);
            }
            //same as above
            else if (this.isLowering === true) {
                let dest = this.getClosestLocOnPlane(this.grass_plane, context, program_state, false);
                this.lowerPlane(this.grass_plane, dest, 20);
                this.grass_plane.shape.copy_onto_graphics_card(context.context);
            }
            //if we are occluding, then draw on the texture, but here we care about yAxis position. The texture is sent to the graphics card later, so not sent now
            else if (this.isOccluding === true){
                let dest = this.getClosestLocOnPlane(this.grass_plane, context, program_state, true);
                this.drawnOnTexture(this.grassOcclusionTexture, this.grass_plane.shape.length, this.grass_plane.shape.width, dest, 13);
            }
            else if (this.placeTree === true) {
                let dest = this.getClosestLocOnPlane(this.grass_plane, context, program_state, true);
                this.shapesArray.push(new Scene_Object(this.shapes.tree, Mat4.translation(dest[0], 6 + dest[1], dest[2]).times(Mat4.scale(1.4,1.4,1.4)), this.materials.tree));
            }
            else if (this.placeRock === true) {
                let dest = this.getClosestLocOnPlane(this.grass_plane, context, program_state, true);
                this.shapesArray.push(new Scene_Object(this.shapes.rock, Mat4.translation(dest[0], dest[1], dest[2]).times(Mat4.scale(1/2, 1/2, 1/2)), this.materials.rock));
            }


            else if(this.solveMaze === true){
                this.pathArr = this.Maze_Solver.solveMaze(this.obstacleArr,this.obstacleArr.length,26,26,7);
                for(let i = 0; i < this.pathArr.length; i++)
                {
                    this.shapesArray.push(new Scene_Object(this.shapes.tree, Mat4.translation(this.pathArr[i][0], 3 + this.pathArr[i][1] , this.pathArr[i][2]), this.materials.plastic));
                }
                this.solveMaze = false;
            }

            
        }

        //to make the grass slowly come back after painted away, just subtract from the red value of the texture every frame
        for(let i = 0; i < 256 * 256 * 4; i++) {
            if (this.grassOcclusionTexture.data[i] > 0){
                this.grassOcclusionTexture.data[i] -= 8;
            }
        }
        this.grassOcclusionTexture.copy_onto_graphics_card(context.context, false);

        let projTransformStorage = program_state.projection_transform;
        let cameraStorage = program_state.camera_inverse;

        this.setFrameBufferForShadows(context, program_state);
        this.render_scene(context, program_state, true, true);

        this.setFrameBufferForDepthPass(context, program_state, projTransformStorage, cameraStorage);
        this.render_scene(context, program_state, true, true, false);

        this.resetFrameBuffer(context, program_state);
        this.render_scene(context, program_state, false);
    }
}