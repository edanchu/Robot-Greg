import {defs, tiny} from './examples/common.js';
const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Matrix, Mat4, Light, Shape, Material, Scene, Shader, Graphics_Card_Object, Texture
} = tiny;

// custom shape class that creates a triangle strip plane with a given length (z wise) and width (x wise) centered around a
// given origin. Each vertex is placed such that there are density number of vertices between each unit distance
// (eg: density 10 means that you'll get 10 vertices between (0,0,0) and (1,0,0))
export class Triangle_Strip_Plane extends Shape{
    constructor(length, width, origin, density){
        super("position", "normal", "texture_coord");
        this.length = length;
        this.width = width;
        this.density = density;
        let denseWidth = width * density;
        let denseLength = length * density;
        // create vertex positions, normals and texture coords. texture coords go from 0,1 in top left to 1,0 in bottom right and are
        // just interpolated by percentage of the way from 0 -> number of desired vertices
        for (let z = 0; z < denseWidth; z++){
            for (let x = 0; x < denseLength; x++){
                this.arrays.position.push(Vector3.create(x/density - length/2 + origin[0] + 1,origin[1],z/density - width/2 + origin[2] + 1));
                this.arrays.texture_coord.push(Vector.create(x/denseLength,1 - (z/denseWidth)));
                //this.arrays.normal.push(Vector3.create(x/density - length/2 + origin[0] + 1,origin[1],z/density - width/2 + origin[2] + 1));
                this.arrays.normal.push(Vector3.create(0,1,0));
            }
        }

        //create the index buffer by connecting points by right hand rule starting by top left, then one under, then one right of the original point and so on
        //in order for the triangle strips to work need to double up on the last index in every row, and the one right after. I can explain why in person
        for (let z = 0; z < denseWidth - 1; z++) {
            if (z > 0) this.indices.push(z * denseLength);
            for (let x = 0; x < denseLength; x++) {
                this.indices.push((z * denseLength) + x, ((z + 1) * denseLength) + x);
            }
            if (z < denseWidth - 2) this.indices.push(((z + 2) * denseLength) - 1);
        }
    }

    //find the closest vertex to a point in a given direction, can specify if to count yAxis (height) difference or not
    closestVertexToRay(origin, direction, yAxis = true){
        let minDistance = 999999999;
        let finalPos;

        //loop through each vertex in the shape and find the closest one. distanceNoDir is the distance between the origin and the vertex you are looking at
        //dest is the destination point made by moving the origin's location by distanceNoDir in the direction's direction
        //distance is the distance between this destination point and the vertex we are checking
        for (let i = 0; i < this.arrays.position.length; i++){
            let distanceNoDir = Vector3.create(origin[0], origin[1], origin[2]).minus(Vector3.create(this.arrays.position[i][0], yAxis ? this.arrays.position[i][1] : 0, this.arrays.position[i][2])).norm();
            let dest = Vector3.create(direction[0], direction[1], direction[2]).times(distanceNoDir).plus(Vector3.create(origin[0], origin[1], origin[2]));
            let distance = Math.abs((dest.minus(Vector3.create(this.arrays.position[i][0], yAxis ? this.arrays.position[i][1] : 0, this.arrays.position[i][2])).norm()));

            if (distance < minDistance){
                minDistance = distance;
                finalPos = i;
            }
        }
        return this.arrays.position[finalPos];
    }

    //updates a given vertex to a new height by changing the position and normal
    updateVertexHeight(x, z, newHeight){
        if ((x + (z * this.width * this.density)) < this.arrays.position.length && (x + (z * this.width * this.density)) >= 0) {
            this.arrays.position[x + (z * this.width * this.density)][1] = newHeight;
            this.arrays.normal[x + (z * this.width * this.density)][1] = newHeight;
        }
    }

    //adds to a given vertex's height (normal and position) up to a specified max value
    addVertexHeight(x, z, newHeight, max = 20){
        if ((x + (z * this.width * this.density)) < this.arrays.position.length && (x + (z * this.width * this.density)) >= 0) {
            if (this.arrays.position[x + (z * this.width * this.density)][1] + newHeight <= max) {
                this.arrays.position[x + (z * this.width * this.density)][1] += newHeight;
                for(let i = -1; i < 2; i++){
                    for(let j = -1; j < 2; j++) {
                        if (((x + i) + ((z + j) * this.width * this.density)) < this.arrays.position.length && ((x + i) + ((z + j) * this.width * this.density)) >= 0){
                            this.arrays.normal[(x + i) + ((z + j) * this.width * this.density)][0] =
                                Math.max(Math.min(this.arrays.normal[(x + i) + ((z + j) * this.width * this.density)][0] + i * 0.001, 0.12),  -0.12);
                            this.arrays.normal[(x + i) + ((z + j) * this.width * this.density)][2] =
                                Math.max(Math.min(this.arrays.normal[(x + i) + ((z + j) * this.width * this.density)][2] + j * 0.001, 0.12), -0.12);
                        }
                    }
                }
            }
            else {
                this.arrays.position[x + (z * this.width * this.density)][1] = max;
            }
        }
    }

    //removes from a given vertex's height (normal and position) down to a specified min value
    removeVertexHeight(x, z, newHeight, min = -10){
        if ((x + (z * this.width * this.density)) < this.arrays.position.length && (x + (z * this.width * this.density)) >= 0) {
            if (this.arrays.position[x + (z * this.width * this.density)][1] - newHeight >= min) {
                this.arrays.position[x + (z * this.width * this.density)][1] -= newHeight;
                for(let i = -1; i < 2; i++){
                    for(let j = -1; j < 2; j++) {
                        if (((x + i) + ((z + j) * this.width * this.density)) < this.arrays.position.length && ((x + i) + ((z + j) * this.width * this.density)) >= 0){
                            this.arrays.normal[(x + i) + ((z + j) * this.width * this.density)][0] =
                                Math.max(Math.min(this.arrays.normal[(x + i) + ((z + j) * this.width * this.density)][0] - i * 0.0015, 0.12),  -0.12);
                            this.arrays.normal[(x + i) + ((z + j) * this.width * this.density)][2] =
                                Math.max(Math.min(this.arrays.normal[(x + i) + ((z + j) * this.width * this.density)][2] - j * 0.0015, 0.12), -0.12);
                        }
                    }
                }
            }
            else {
                this.arrays.position[x + (z * this.width * this.density)][1] = min;
                //this.arrays.normal[x + (z * this.width * this.density)][1] = min;
            }
        }
    }
}

//custom texture class that uses a software defined array as input instead of the html IMAGE object that the default class uses
//need to pass in the length and width of the data. If no data passed in, will create a black texture of the given size
//most of this is documented in the tiny-graphics.js file, so I will just comment the changes I made from that
export class Dynamic_Texture extends Graphics_Card_Object {
    constructor(length, width, data = null, min_filter = "LINEAR_MIPMAP_LINEAR") {
        super();
        this.length = length;
        this.width = width;
        //this is a texture filtering thing, and I don't know much about it. its from the default texture class in tiny graphics
        this.min_filter = min_filter;

        //copy in the data, or make new data if none is passed in
        this.data = data;
        if (this.data == null) {
            this.data = [];
            for (let i = 0; i < this.length * this.width; i++) {
                this.data.push(0, 0, 0, 255);
            }
        }
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
        //the original version of this function had a queried a ready flag to see if the texture had been loaded
        //we are providing our own data in software, so I removed both the flag and the test
        //if passing multiple textures in to a shader, set texture unit so that you know which texture is which in the shader
        const gpu_instance = super.activate(context);
        context.activeTexture(context["TEXTURE" + texture_unit]);
        context.bindTexture(context.TEXTURE_2D, gpu_instance.texture_buffer_pointer);
    }
}

//This class is mostly the movement and mouse controls from common.js but with modifications for our program.
//I added an isPainting variable that toggles on m1 being held down, and changed panning to use m2. Furthermore, I added mouse position to the live readout
//The vast majority of this is documented in common.js, so I won't add comments for what I didn't change
export class Custom_Movement_Controls extends defs.Movement_Controls{

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
        //this.live_string(box => box.textContent = "- Mouse Pos: " + (this.mouse.from_center));
        //this.new_line();
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
        //this.key_triggered_button("Go to world origin", ["r"], () => {
        //    this.matrix().set_identity(4, 4);
        //    this.inverse().set_identity(4, 4)
        //}, "#8B8885");
        //this.new_line();

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

export class Buffered_Texture extends tiny.Graphics_Card_Object {
    // **Texture** wraps a pointer to a new texture image where
    // it is stored in GPU memory, along with a new HTML image object.
    // This class initially copies the image to the GPU buffers,
    // optionally generating mip maps of it and storing them there too.
    constructor(texture_buffer_pointer) {
        super();
        Object.assign(this, {texture_buffer_pointer});
        this.ready = true;
        this.texture_buffer_pointer = texture_buffer_pointer;
    }

    copy_onto_graphics_card(context, need_initial_settings = true) {
        // copy_onto_graphics_card():  Called automatically as needed to load the
        // texture image onto one of your GPU contexts for its first time.

        // Define what this object should store in each new WebGL Context:
        const initial_gpu_representation = {texture_buffer_pointer: undefined};
        // Our object might need to register to multiple GPU contexts in the case of
        // multiple drawing areas.  If this is a new GPU context for this object,
        // copy the object to the GPU.  Otherwise, this object already has been
        // copied over, so get a pointer to the existing instance.
        const gpu_instance = super.copy_onto_graphics_card(context, initial_gpu_representation);

        if (!gpu_instance.texture_buffer_pointer) gpu_instance.texture_buffer_pointer = this.texture_buffer_pointer;
        return gpu_instance;
    }

    activate(context, texture_unit = 0) {
        // activate(): Selects this Texture in GPU memory so the next shape draws using it.
        // Optionally select a texture unit in case you're using a shader with many samplers.
        // Terminate draw requests until the image file is actually loaded over the network:
        if (!this.ready)
            return;
        const gpu_instance = super.activate(context);
        context.activeTexture(context["TEXTURE" + texture_unit]);
        context.bindTexture(context.TEXTURE_2D, this.texture_buffer_pointer);
    }
}

export class Scene_Object{
    constructor(shape, transform, material, renderArgs = "TRIANGLES") {
        this.shape = shape;
        this.transform = transform;
        this.material = material;
        this.renderArgs = renderArgs;
    }

    drawObject(context, program_state){
        this.shape.draw(context, program_state, this.transform, this.material, this.renderArgs);
    }

    drawOverrideMaterial(context, program_state, overrideMat){
        this.shape.draw(context, program_state, this.transform, overrideMat, this.renderArgs);
    }
}