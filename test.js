import {defs, tiny} from './examples/common.js';

const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Matrix, Mat4, Light, Shape, Material, Scene, Shader, Graphics_Card_Object, Texture
} = tiny;

class Cube extends Shape {
    constructor() {
        super("position", "normal",);
        // Loop 3 times (for each axis), and inside loop twice (for opposing cube sides):
        this.arrays.position = Vector3.cast(
            [-1, -1, -1], [1, -1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [1, 1, 1], [-1, 1, 1],
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, 1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, -1], [-1, -1, -1], [1, 1, -1], [-1, 1, -1]);
        this.arrays.normal = Vector3.cast(
            [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]);
        // Arrange the vertices into a square shape in texture space too:
        this.indices.push(0, 1, 2, 1, 3, 2, 4, 5, 6, 5, 7, 6, 8, 9, 10, 9, 11, 10, 12, 13,
            14, 13, 15, 14, 16, 17, 18, 17, 19, 18, 20, 21, 22, 21, 23, 22);
    }
}

class Cube_Outline extends Shape {
    constructor() {
        super("position", "color");
        //  TODO (Requirement 5).
        // When a set of lines is used in graphics, you should think of the list entries as
        // broken down into pairs; each pair of vertices will be drawn as a line segment.
        // Note: since the outline is rendered with Basic_shader, you need to redefine the position and color of each vertex

        this.arrays.position = Vector3.cast(
            [-1, -1, -1], [1, -1, -1],  [-1, -1,-1], [-1, 1,-1], [-1, -1,-1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1],
            [1, -1, -1], [1, 1, -1],  [-1, 1,-1], [-1, 1, 1], [1, 1,-1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
            [-1, -1, 1], [-1, 1, 1],  [-1, -1, 1], [1, -1,1], [-1, 1,1], [1, 1, 1], [1, 1, 1], [1, -1, 1]);

        this.arrays.color.push(color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1),
            color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1),
            color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1),
            color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1), color(1,1,1,1));

        this.indices = false;

    }
}

class Cube_Single_Strip extends Shape {
    constructor() {
        super("position", "normal");

        this.arrays.position = Vector3.cast(
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1],
            [1, 1, 1]);

        this.arrays.normal = Vector3.cast(
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1],
            [1, 1, 1]);

        this.indices.push(7,6,5,4,0,5,1,7,3,6,2,4,0,2,1,3);

    }
}

// custom shape class that creates a triangle strip plane with a given length (z wise) and width (x wise) centered around a
// given origin. Each vertex is placed such that there are density number of vertices between each unit distance
// (eg: density 10 means that youll get 10 vertices between (0,0,0) and (1,0,0))
class Triangle_Strip_Plane extends Shape{
    constructor(length, width, origin, density){
        super("position", "normal", "texture_coord");
        let denseWidth = width * density;
        let denseLength = length * density;
        //create vertex positions and texture coords. texture coords go from 0,1 in top left to 1,0 in bottom right and are
        // just normalized by percentage of the way from 0 -> number of wanted vertices
        for (let z = 0; z < denseWidth; z++){
            for (let x = 0; x < denseLength; x++){
                this.arrays.position.push(Vector3.create(x/density - length/2 + origin[0] + 1,origin[1],z/density - width/2 + origin[2] + 1));
                this.arrays.texture_coord.push(Vector.create(x/denseLength,1 - (z/denseWidth)));
            }
        }
        //copy position array into the normals array. I think there is probably a better way to do this, but I'm not sure since I'm crap at javascript
        this.arrays.normal.push.apply(this.arrays.normal, this.arrays.position);

        //create the index buffer by connecting points by right hand rule starting by top left, then one under, then one left of the original point
        //in order for the triangle strips to work need to double up on the last index in every row, and the one right after. I can explain why in person
        for (let z = 0; z < denseWidth - 1; z++) {
            if (z > 0) this.indices.push(z * denseLength);
            for (let x = 0; x < denseLength; x++) {
                this.indices.push((z * denseLength) + x, ((z + 1) * denseLength) + x);
            }
            if (z < denseWidth - 2) this.indices.push(((z + 2) * denseLength) - 1);
        }
    }

    //find the closest vertex to a point in a given direction
    intersection(origin, direction){
        let minDistance = 999999999;
        let finalPos;

        //loop through each vertex in the shape and find the closest one. distanceVec is the distance between the origin and the vertex you are looking at
        //dest is the destination point made by moving the origin's location by distanceVec in the direction's direction
        //distance is the distance between this destination point and the vertex
        for (let i = 0; i < this.arrays.position.length; i++){
            let distanceVec = Vector3.create(origin[0], origin[1], origin[2]).minus(Vector3.create(this.arrays.position[i][0], this.arrays.position[i][1], this.arrays.position[i][2]));
            let dest = (Vector3.create(direction[0], direction[1], direction[2]).times(distanceVec.norm())).plus(Vector3.create(origin[0], origin[1], origin[2]));
            let distance = Math.abs((dest.minus(Vector3.create(this.arrays.position[i][0], this.arrays.position[i][1], this.arrays.position[i][2])).norm()));

            if (distance < minDistance){
                minDistance = distance;
                finalPos = Vector3.create(this.arrays.position[i][0], this.arrays.position[i][1], this.arrays.position[i][2]);
            }
        }

        return finalPos;

    }
}

// custom shader class that takes in a texture and uses that texture's red value to offset the vertex's y position
// paints the texture onto the shape as color. Eventually I think maybe we can pass in 2 textures, one as a heightmap
// and one as the actual texture we want to be displayed. To learn how these work check out the examples in common.js
class Offset_shader extends Shader {
    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform],
            PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniform4fv(gpu_addresses.color, material.color);

        //upload the texture at the gpu's 0 index. The next line is what sets the texture we want to that index (at least, as far as I can tell)
        context.uniform1i(gpu_addresses.texture, 0);
        material.texture.activate(context);
    }

    //glsl code that goes into both the vertex and fragment shaders. the code here gives them both the texture we uploaded to the gpu
    shared_glsl_code() {
        return `precision mediump float;
                uniform sampler2D texture;
                varying vec2 f_tex_coord;   
            `;
    }

    //sets each vertex's position to the its position + the red value of our texture
    vertex_glsl_code() {
        return this.shared_glsl_code() + `
                attribute vec3 position;                         
                uniform mat4 projection_camera_model_transform;
                attribute vec2 texture_coord; 
                
                void main(){
                    vec4 tex_color = texture2D(texture, texture_coord);
                    gl_Position = projection_camera_model_transform * vec4( position.x, position.y + tex_color.r, position.z, 1.0 );
                    f_tex_coord = texture_coord;
                }`;
    }

    //sets each pixel's color
    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                uniform vec4 color;
                void main(){
                    vec4 tex_color = texture2D(texture, f_tex_coord);
                    gl_FragColor = tex_color;
                    //gl_FragColor = color;
                }`;
    }
}

//custom texture class that uses a software defined array as input instead of the html IMAGE object that the default class uses
//need to pass in the length and width of the data as well as the data itself
//most of this is documented in the tiny-graphics.js file, so I will just comment the changes I made from that
class Custom_Texture extends Graphics_Card_Object {
    //im not sure why this assigns object properties this way, but I just kept what they did in tiny.js and added the length, width, etc designations
    constructor(length, width, data, min_filter = "LINEAR_MIPMAP_LINEAR") {
        super();
        Object.assign(this, {length, width, data, min_filter});
        //normally there would be declarations of an html IMAGE object here. We don't want to use a source file and want to
        //pass in our own data, so I removed that
    }

    copy_onto_graphics_card(context, need_initial_settings = true) {
        const initial_gpu_representation = {texture_buffer_pointer: undefined};

        const gpu_instance = super.copy_onto_graphics_card(context, initial_gpu_representation);

        if (!gpu_instance.texture_buffer_pointer) gpu_instance.texture_buffer_pointer = context.createTexture();

        const gl = context;
        gl.bindTexture(gl.TEXTURE_2D, gpu_instance.texture_buffer_pointer);

        if (need_initial_settings) {
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl[this.min_filter]);
        }
        //this converts our data array which is 4 8bit color values per pixel into the format that opengl uses and clamps in case our values are too high/low
        let imageData = new Uint8ClampedArray(this.data);
        //creates the texture based on our data
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.length, 0, gl.RGBA, gl.UNSIGNED_BYTE, imageData);
        if (this.min_filter === "LINEAR_MIPMAP_LINEAR")
            gl.generateMipmap(gl.TEXTURE_2D);

        return gpu_instance;
    }

    activate(context, texture_unit = 0) {
        //the original version of this function had a queried a ready flag to see if the texture had been loaded from disk
        //we are providing our own data in software, so I removed both the flag and the test
        const gpu_instance = super.activate(context);
        context.activeTexture(context["TEXTURE" + texture_unit]);
        context.bindTexture(context.TEXTURE_2D, gpu_instance.texture_buffer_pointer);
    }
}

//This class is mostly the movement and mouse controls from common.js but with modifications for our program.
//I added an isPainting variable that toggles on m1 being held down, and changed panning to use m2. Furthermore, I added mouse position to the live readout
//The vast majority of this is documented in common.js, so I won't add comments for what I didn't change
class Custom_Movement_Controls extends defs.Movement_Controls{

    constructor(){
        super();
        //a mouse object that we can query from our main program to get location and pressedness
        this.mouse = {"from_center": vec(0, 0), "isPainting": false};
    }

    add_mouse_controls(canvas) {
        const mouse_position = (e, rect = canvas.getBoundingClientRect()) =>
            vec(e.clientX - (rect.left + rect.right) / 2, e.clientY - (rect.bottom + rect.top) / 2);
        document.addEventListener("mouseup", e => {
            //button 2 is rmb, which is used for panning
            if (e.button === 2) {
                this.mouse.anchor = undefined;
            }
            //button 0 is lmb used for painting
            else if (e.button === 0) {
                this.mouse.isPainting = false;
            }
        });
        canvas.addEventListener("mousedown", e => {
            e.preventDefault();
            if (e.button === 2) {
                this.mouse.anchor = mouse_position(e);
            }
            else if (e.button === 0) {
                this.mouse.isPainting = true;
            }
        });
        canvas.addEventListener("mousemove", e => {
            e.preventDefault();
            this.mouse.from_center = mouse_position(e);
        });
        canvas.addEventListener("mouseout", e => {
            if (!this.mouse.anchor) this.mouse.from_center.scale_by(0)
        });
        //this line is used so that right click won't bring up the context menu when it is clicked on our 3d area.
        //I got this from the documentation, and I'm not sure sure how it works
        canvas.oncontextmenu = function(e) {e.preventDefault()};
    }

    make_control_panel() {
        this.control_panel.innerHTML += "Click and drag right mouse button to spin your viewpoint around it.<br>";
        this.live_string(box => box.textContent = "- Position: " + this.pos[0].toFixed(2) + ", " + this.pos[1].toFixed(2)
            + ", " + this.pos[2].toFixed(2));
        this.new_line();
        this.live_string(box => box.textContent = "- Facing: " + ((this.z_axis[0] > 0 ? "West " : "East ")
            + (this.z_axis[1] > 0 ? "Down " : "Up ") + (this.z_axis[2] > 0 ? "North" : "South")));
        this.new_line();
        //added a live readout of the mouse's position from the center of the screen in pixel coordinates
        this.live_string(box => box.textContent = "- Mouse Pos: " + (this.mouse.from_center));
        this.new_line();
        this.new_line();

        this.key_triggered_button("Up", [" "], () => this.thrust[1] = -1, undefined, () => this.thrust[1] = 0);
        this.key_triggered_button("Forward", ["w"], () => this.thrust[2] = 1, undefined, () => this.thrust[2] = 0);
        this.new_line();
        this.key_triggered_button("Left", ["a"], () => this.thrust[0] = 1, undefined, () => this.thrust[0] = 0);
        this.key_triggered_button("Back", ["s"], () => this.thrust[2] = -1, undefined, () => this.thrust[2] = 0);
        this.key_triggered_button("Right", ["d"], () => this.thrust[0] = -1, undefined, () => this.thrust[0] = 0);
        this.new_line();
        this.key_triggered_button("Down", ["z"], () => this.thrust[1] = 1, undefined, () => this.thrust[1] = 0);

        const speed_controls = this.control_panel.appendChild(document.createElement("span"));
        speed_controls.style.margin = "30px";
        this.key_triggered_button("-", ["o"], () =>
            this.speed_multiplier /= 1.2, undefined, undefined, undefined, speed_controls);
        this.live_string(box => {
            box.textContent = "Speed: " + this.speed_multiplier.toFixed(2)
        }, speed_controls);
        this.key_triggered_button("+", ["p"], () =>
            this.speed_multiplier *= 1.2, undefined, undefined, undefined, speed_controls);
        this.new_line();
        this.key_triggered_button("Roll left", [","], () => this.roll = 1, undefined, () => this.roll = 0);
        this.key_triggered_button("Roll right", ["."], () => this.roll = -1, undefined, () => this.roll = 0);
        this.new_line();
        this.key_triggered_button("(Un)freeze mouse look around", ["f"], () => this.look_around_locked ^= 1, "#8B8885");
        this.new_line();
        this.key_triggered_button("Go to world origin", ["r"], () => {
            this.matrix().set_identity(4, 4);
            this.inverse().set_identity(4, 4)
        }, "#8B8885");
        this.new_line();

        this.key_triggered_button("Look at origin from front", ["1"], () => {
            this.inverse().set(Mat4.look_at(vec3(0, 0, 10), vec3(0, 0, 0), vec3(0, 1, 0)));
            this.matrix().set(Mat4.inverse(this.inverse()));
        }, "#8B8885");
        this.new_line();
        this.key_triggered_button("from right", ["2"], () => {
            this.inverse().set(Mat4.look_at(vec3(10, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0)));
            this.matrix().set(Mat4.inverse(this.inverse()));
        }, "#8B8885");
        this.key_triggered_button("from rear", ["3"], () => {
            this.inverse().set(Mat4.look_at(vec3(0, 0, -10), vec3(0, 0, 0), vec3(0, 1, 0)));
            this.matrix().set(Mat4.inverse(this.inverse()));
        }, "#8B8885");
        this.key_triggered_button("from left", ["4"], () => {
            this.inverse().set(Mat4.look_at(vec3(-10, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0)));
            this.matrix().set(Mat4.inverse(this.inverse()));
        }, "#8B8885");
        this.new_line();
        this.key_triggered_button("Attach to global camera", ["Shift", "R"],
            () => {
                this.will_take_over_graphics_state = true
            }, "#8B8885");
        this.new_line();
    }
}

class Base_Scene extends Scene {
    constructor() {
        super();

        //define how long and wide we want our plane to be. the density gives extra resolution by making more triangles in the same amount of space
        this.planeWidth = 20;
        this.planeLength = 20;
        this.density = 7;

        //declare a 256 by 256 array that will be used as our texture to store height data. for now it also has values in the other color channels
        //since it is being used to color the plane as well, but that will change. Each color is from 0 to 255
        // (this isn't actually constrained since js has no types and I don't know how to constrain that from here)
        this.offsetsWidth = 256;
        this.offsetsLength = 256;
        this.offsets = [];
        for (let i = 0; i < this.offsetsWidth * this.offsetsLength; i++){
            this.offsets.push(0, 0, 255, 255);
        }

        //creates our custom texture based on the heightmap data we just created
        this.texture = new Custom_Texture(this.offsetsLength, this.offsetsWidth, this.offsets);

        this.shapes = {
            'square' : new defs.Square(),
            'cube': new Cube(),
            'outline': new Cube_Outline(),
            'single_strip' : new Cube_Single_Strip(),
            //creates our custom plane with the previously declared length, width, density, and an origin of (0,0,0)
            'plane' : new Triangle_Strip_Plane(this.planeLength,this.planeWidth, Vector3.create(0,0,0), this.density),
            'axis' : new defs.Axis_Arrows()
        };

        this.materials = {
            plastic: new Material(new defs.Phong_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            //creates a material based on our texture. For now it also takes a color variable, but I think we can get rid of that at some point
            offset: new Material(new Offset_shader(), {color: hex_color("#2b3b86"), texture: this.texture}),
        };
        this.white = new Material(new defs.Basic_Shader());
    }

    display(context, program_state) {
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new Custom_Movement_Controls());
            //sets the camera at location (6,7,25), looking at the origin
            program_state.set_camera(Mat4.look_at(vec3(6, 7, 25), vec3(0, 0, 0), vec3(0, 1, 0)))
        }
        program_state.projection_transform = Mat4.perspective(
            Math.PI/4 , context.width / context.height, 1, 100);

        const light_position = vec4(0, 5, 5, 1);
        program_state.lights = [new Light(light_position, color(1, 1, 1, 1), 1000)];
    }
}

export class Test extends Base_Scene {
    constructor() {
        super();
    }

    make_control_panel(context) {
        this.key_triggered_button("Placeholder", ["c"], () => 1);
    }

    //helper function to get the location of the closest vertex on our plane to where the mouse is pointing
    getClosestLocOnPlane(context, program_state) {
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
        //this is the world space coordinates of the near point. We divide by the last homogenous value because of the perspective transform. I am not 100% sure why we need this though
        //         //I need to look more into this and as the TA as well
        worldSpaceNear = Vector.create(worldSpaceNear[0] / worldSpaceNear[3], worldSpaceNear[1] / worldSpaceNear[3], worldSpaceNear[2] / worldSpaceNear[3], 1);

        let mousePointFar = Vector.create(mousePosPercent[0], mousePosPercent[1], 1, 1);
        let worldSpaceFar = transformMatrix.times(mousePointFar);
        //this is the world space coordinates of the far point. We divide by the last homogenous value because of the perspective transform. I am not 100% sure why we need this though
        //I need to look more into this and as the TA as well
        worldSpaceFar = Vector.create(worldSpaceFar[0] / worldSpaceFar[3], worldSpaceFar[1] / worldSpaceFar[3], worldSpaceFar[2] / worldSpaceFar[3], 1);

        //this calls the intersection function of the plane shape to find which vertex is nearest to the mouse click. the function takes
        //our mouse's location as the origin, and a vector direction to the world space location of the far coordinate
        return this.shapes.plane.intersection(worldSpaceNear, worldSpaceFar.minus(worldSpaceNear).normalized());
    }

    //this function draws onto the heightmap texture when given a location to draw (in world space coordinates) and a brush radius.
    //eventually I intend for this to create a circle of that radius, but for now its just a square of size 10
    drawnOnTexture(location, brushRadius) {
        //this find what percentage to the edge the location given is. I currently treat the texture as taking up the same world space size as the size of the plane
        let textureLocPercent = Vector.create((location[0]-1) / (this.planeWidth / 2), (location[2]-1) / (this.planeLength / 2));
        //using the percentage we can find what z and x coordinates that would be given 128 rows to the right and down from the origin
        let textureLoc = Vector.create(Math.ceil(textureLocPercent[0] * 128) + 128, Math.ceil(textureLocPercent[1] * 128) + 128);
        //translate from the z and x coordinates into the texture and set the value to 255. ideally Id want it to be a circle with decreasing height from the center
        //the *4 is because each "pixel" has r,g,b,a
        for (let z = textureLoc[1]; z < textureLoc[1] + 10; z++) {
            for (let x = textureLoc[0]; x < textureLoc[0] + 10; x++){
                this.offsets[(x*4) + (z * 4 * 256)] = 255;
            }
        }
    }

    display(context, program_state) {
        super.display(context, program_state);

        //this was what I had before for the sin wave offsets
        //for (let z = 0; z < this.offsetsWidth * 4; z+= 4){
        //    for (let x = 0; x < this.offsetsLength * 4; x+= 4){
        //       // this.offsets[x+z*this.offsetsWidth] = 128 * (Math.sin(x/20 - program_state.animation_time/1000) + 1);
        //        this.offsets[x+(z*this.offsetsWidth)+1] = 0;
        //        this.offsets[x+(z*this.offsetsWidth)+2] = 255;
        //        this.offsets[x+(z*this.offsetsWidth)+3] = 255;
        //    }
        //}

        //if the mouse is held down and we are trying to paint on the canvas
        if (context.scratchpad.controls.mouse.isPainting) {
            //find the location we are trying to paint at
            let dest = this.getClosestLocOnPlane(context, program_state);
            //paint on the texture with a brush radius of 10 (the brush radius doesn't actually do anything yet)
            this.drawnOnTexture(dest, 10);
            //send the newly updated texture to the gpu
            this.texture.copy_onto_graphics_card(context.context, false);
        }

        //draw the plane and axis (axis was just so I could see if it is actually centered, I should honestly just remove it)
        let model_transform = Mat4.identity();
        this.shapes.plane.draw(context, program_state, model_transform, this.materials.offset, "TRIANGLE_STRIP");
        this.shapes.axis.draw(context, program_state, model_transform, this.materials.plastic);
    }
}