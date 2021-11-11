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

export class Maze_Solver {

    printSolution(sol, N, density) {
        let arr = [];
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++)
                if (sol[i][j] === 1)
                    arr.push([i / density - 13, 0, j / density - 13]);
        }
        return arr;
    }


    inBounds(maze, N, x, y) {
        // if (x, y outside maze) return false
        return (x >= 0 && x < N && y >= 0 && y < N );
    }

    solveMaze(maze, N, len, wid, density) {
        let sol = new Array(N);
        for (let i = 0; i < N; i++) {
            sol[i] = new Array(N);
            for (let j = 0; j < N; j++) {
                sol[i][j] = 0;
            }
        }

        if (this.solveMazeUtil(maze, 0, 0, sol, N) === false) {
            document.write("Solution doesn't exist");
            return false;
        }

        return this.printSolution(sol, N, density);
    }

    solveMazeUtil(maze, x, y, sol, N) {
        // if (x, y is goal) return true
        if (x === N - 1 && y === N - 1
            && maze[x][y] === 1) {
            sol[x][y] = 1;
            return true;
        }


        if (this.inBounds(maze,N, x, y) === true) {

            if (sol[x][y] === 1)
                return false;

            sol[x][y] = 1;

            if (this.solveMazeUtil(maze, x + 1, y, sol, N))
                return true;


            if (this.solveMazeUtil(maze, x, y + 1, sol, N))
                return true;


            if (this.solveMazeUtil(maze, x - 1, y, sol, N))
                return true;

            if (this.solveMazeUtil(maze, x, y - 1, sol, N))
                return true;

            sol[x][y] = 0;
            return false;
        }

        return false;
    }


}

export class Shape_From_File extends Shape {                                   // **Shape_From_File** is a versatile standalone Shape that imports
                                                                               // all its arrays' data from an .obj 3D model file.
    constructor(filename) {
        super("position", "normal", "texture_coord", "tangent");
        // Begin downloading the mesh. Once that completes, return
        // control to our parse_into_mesh function.
        this.load_file(filename);
    }
    
    load_file(filename) {                             // Request the external file and wait for it to load.
        // Failure mode:  Loads an empty shape.
        return fetch(filename)
            .then(response => {
                if (response.ok) return Promise.resolve(response.text())
                else return Promise.reject(response.status)
            })
            .then(obj_file_contents => this.parse_into_mesh(obj_file_contents))
            .catch(error => {
                this.copy_onto_graphics_card(this.gl);
            })
    }
    
    parse_into_mesh(data) {                           // Adapted from the "webgl-obj-loader.js" library found online:
        var verts = [], vertNormals = [], textures = [], unpacked = {};
    
        unpacked.verts = [];
        unpacked.norms = [];
        unpacked.textures = [];
        unpacked.hashindices = {};
        unpacked.indices = [];
        unpacked.index = 0;
    
        var lines = data.split('\n');
    
        var VERTEX_RE = /^v\s/;
        var NORMAL_RE = /^vn\s/;
        var TEXTURE_RE = /^vt\s/;
        var FACE_RE = /^f\s/;
        var WHITESPACE_RE = /\s+/;
    
        for (var i = 0; i < lines.length; i++) {
            var line = lines[i].trim();
            var elements = line.split(WHITESPACE_RE);
            elements.shift();
        
            if (VERTEX_RE.test(line)) verts.push.apply(verts, elements);
            else if (NORMAL_RE.test(line)) vertNormals.push.apply(vertNormals, elements);
            else if (TEXTURE_RE.test(line)) textures.push.apply(textures, elements);
            else if (FACE_RE.test(line)) {
                var quad = false;
                for (var j = 0, eleLen = elements.length; j < eleLen; j++) {
                    if (j === 3 && !quad) {
                        j = 2;
                        quad = true;
                    }
                    if (elements[j] in unpacked.hashindices)
                        unpacked.indices.push(unpacked.hashindices[elements[j]]);
                    else {
                        var vertex = elements[j].split('/');
                    
                        unpacked.verts.push(+verts[(vertex[0] - 1) * 3 + 0]);
                        unpacked.verts.push(+verts[(vertex[0] - 1) * 3 + 1]);
                        unpacked.verts.push(+verts[(vertex[0] - 1) * 3 + 2]);
                    
                        if (textures.length) {
                            unpacked.textures.push(+textures[((vertex[1] - 1) || vertex[0]) * 2 + 0]);
                            unpacked.textures.push(+textures[((vertex[1] - 1) || vertex[0]) * 2 + 1]);
                        }
                    
                        unpacked.norms.push(+vertNormals[((vertex[2] - 1) || vertex[0]) * 3 + 0]);
                        unpacked.norms.push(+vertNormals[((vertex[2] - 1) || vertex[0]) * 3 + 1]);
                        unpacked.norms.push(+vertNormals[((vertex[2] - 1) || vertex[0]) * 3 + 2]);
                    
                        unpacked.hashindices[elements[j]] = unpacked.index;
                        unpacked.indices.push(unpacked.index);
                        unpacked.index += 1;
                    }
                    if (j === 3 && quad) unpacked.indices.push(unpacked.hashindices[elements[0]]);
                }
            }
        }
        {
            const {verts, norms, textures} = unpacked;
            for (var j = 0; j < verts.length / 3; j++) {
                this.arrays.position.push(vec3(verts[3 * j], verts[3 * j + 1], verts[3 * j + 2]));
                this.arrays.normal.push(vec3(norms[3 * j], norms[3 * j + 1], norms[3 * j + 2]));
                this.arrays.texture_coord.push(vec(textures[2 * j], textures[2 * j + 1]));
            }
            this.indices = unpacked.indices;
        }

        let tangents = [];
        let bitangents = [];

        let triCount = this.indices.length;
        for(let i = 0; i< triCount; i+=3){
            let i0 = this.indices[i];
            let i1 = this.indices[i+1];
            let i2 = this.indices[i+2];

            let pos0 = this.arrays.position[i0];
            let pos1 = this.arrays.position[i1];
            let pos2 = this.arrays.position[i2];

            let tex0 = this.arrays.texture_coord[i0];
            let tex1 = this.arrays.texture_coord[i1];
            let tex2 = this.arrays.texture_coord[i2];

            let edge1 = pos1.minus(pos0);
            let edge2 = pos2.minus(pos0);

            let uv1 = tex1.minus(tex0);
            let uv2 = tex2.minus(tex0);

            let r = 1.0 / ((uv1[0] * uv2[1]) - (uv1[1] * uv2[0]));
            if (r === Infinity)
                r = 0;
            
            let tangent = Vector3.create((((edge1[0] * uv2[1]) - (edge2[0] * uv1[1]))* r),
                (((edge1[1] * uv2[1]) - (edge2[1] * uv1[1]))* r),
                (((edge1[2] * uv2[1]) - (edge2[2] * uv1[1]))* r));

            let bitangent = Vector3.create((((edge1[0] * uv2[0]) - (edge2[0] * uv1[0]))* r),
                (((edge1[1] * uv2[0]) - (edge2[1] * uv1[0]))* r),
                (((edge1[2] * uv2[0]) - (edge2[2] * uv1[0]))* r));
            
            if (tangents[i0] == null)
                tangents[i0] = tangent;
            else
                tangents[i0] = tangents[i0].plus(tangent);
            
            if (tangents[i1] == null)
                tangents[i1] = tangent;
            else
                tangents[i1] = tangents[i1].plus(tangent);
            
            if (tangents[i2] == null)
                tangents[i2] = tangent;
            else
                tangents[i2] = tangents[i2].plus(tangent);
    
            
            if (bitangents[i0] == null)
                bitangents[i0] = bitangent;
            else
                bitangents[i0] = bitangents[i0].plus(bitangent);
            
            if (bitangents[i1] == null)
                bitangents[i1] = bitangent;
            else
                bitangents[i1] = bitangents[i1].plus(bitangent);
            
            if (bitangents[i2] == null)
                bitangents[i2] = bitangent;
            else
                bitangents[i2] = bitangents[i2].plus(bitangent);
        }

        for (let i = 0; i < this.arrays.position.length; i++){
            let n = this.arrays.normal[i];
            let t0 = tangents[i];
            let t1 = bitangents[i];
            let tangent = t0.minus(n.times(n.dot(t0))).normalized();
            let cross = n.cross(t0);
            let w = cross.dot(t1) < 0 ? -1.0:1.0;
            this.arrays.tangent.push(tangent);
        }
        
        
        this.normalize_positions(false);
        this.ready = true;
    }
    
    draw(context, program_state, model_transform, material) {               // draw(): Same as always for shapes, but cancel all
        // attempts to draw the shape before it loads:
        if (this.ready)
            super.draw(context, program_state, model_transform, material);
    }
}