import {defs, tiny} from './examples/common.js';

const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Matrix, Mat4, Light, Shape, Material, Scene, Shader, Graphics_Card_Object, Texture
} = tiny;

let globDir;
let globOrig;

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

class Triangle_Strip_Plane extends Shape{
    constructor(length, width, origin, density){
        super("position", "normal", "texture_coord");
        let denseWidth = width * density;
        let denseLength = length * density;
        for (let z = 0; z < denseWidth; z++){
            for (let x = 0; x < denseLength; x++){
                this.arrays.position.push(Vector3.create(x/density - length/2 + origin[0] + 1,origin[1],z/density - width/2 + origin[2] + 1));
                this.arrays.texture_coord.push(Vector.create(x/denseLength,1 - (z/denseWidth)));
            }
        }
        this.arrays.normal.push.apply(this.arrays.normal, this.arrays.position);

        for (let z = 0; z < denseWidth - 1; z++) {
            if (z > 0) this.indices.push(z * denseLength);
            for (let x = 0; x < denseLength; x++) {
                this.indices.push((z * denseLength) + x, ((z + 1) * denseLength) + x);
            }
            if (z < denseWidth - 2) this.indices.push(((z + 2) * denseLength) - 1);
        }
    }

    intersection(origin, direction){
        let minDistance = 999999999;
        let vertexNum = -1;
        let finalDest;

        if(!origin){
            origin = Vector3.create(0,0,0);
            direction = Vector3.create(0,1,0);
        }

        for (let i = 0; i < this.arrays.position.length; i++){
            let distanceVec = Vector3.create(origin[0], -origin[1], origin[2]).minus(Vector3.create(this.arrays.position[i][0], this.arrays.position[i][1], this.arrays.position[i][2]));
            let dest = (direction.times(distanceVec.norm())).plus(Vector3.create(origin[0], origin[1], origin[2]));
            let distance = Math.abs((dest.minus(Vector3.create(this.arrays.position[i][0], this.arrays.position[i][1], this.arrays.position[i][2])).norm()));

            if (distance < minDistance){
                minDistance = distance;
                vertexNum = i;
                finalDest = Vector3.create(this.arrays.position[i][0], this.arrays.position[i][1], this.arrays.position[i][2]);
            }
        }

        if(minDistance !== 999999999){
            //return vertexNum;
            return [finalDest, vertexNum];
        }

        return null;
    }
}

class Offset_shader extends Shader {
    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform],
            PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniform4fv(gpu_addresses.color, material.color);

        context.uniform1i(gpu_addresses.texture, 0);
        material.texture.activate(context);
    }

    shared_glsl_code() {
        return `precision mediump float;
                uniform sampler2D texture;
                varying vec2 f_tex_coord;   
            `;
    }

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

class Custom_Texture extends Graphics_Card_Object {
    constructor(length, width, data, min_filter = "LINEAR_MIPMAP_LINEAR") {
        super();
        Object.assign(this, {length, width, data, min_filter});
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

        let imageData = new Uint8ClampedArray(this.data);

        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.length, 0, gl.RGBA, gl.UNSIGNED_BYTE, imageData);
        if (this.min_filter === "LINEAR_MIPMAP_LINEAR")
            gl.generateMipmap(gl.TEXTURE_2D);

        return gpu_instance;
    }

    activate(context, texture_unit = 0) {
        const gpu_instance = super.activate(context);
        context.activeTexture(context["TEXTURE" + texture_unit]);
        context.bindTexture(context.TEXTURE_2D, gpu_instance.texture_buffer_pointer);
    }
}

class Custom_Movement_Controls extends defs.Movement_Controls{

    constructor(colors, length, width){
        super();
        this.mouse = {"from_center": vec(0, 0)};
        this.colors = colors;
        this.length = length;
        this.width = width;
    }

    add_mouse_controls(canvas) {
        // add_mouse_controls():  Attach HTML mouse events to the drawing canvas.
        // First, measure mouse steering, for rotating the flyaround camera:
        const mouse_position = (e, rect = canvas.getBoundingClientRect()) =>
            vec(e.clientX - (rect.left + rect.right) / 2, e.clientY - (rect.bottom + rect.top) / 2);
        // Set up mouse response.  The last one stops us from reacting if the mouse leaves the canvas:
        document.addEventListener("mouseup", e => {
            if (e.button === 2) {
                this.mouse.anchor = undefined;
            }
            else if (e.button === 0) {
                this.mouse.isPainting = undefined;
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
            globOrig = Vector3.create(this.pos[0] + this.mouse.from_center[0]/20, this.pos[1] + this.mouse.from_center[1]/14, this.pos[2]);
            globDir = this.z_axis;
        });
        canvas.addEventListener("mouseout", e => {
            if (!this.mouse.anchor) this.mouse.from_center.scale_by(0)
        });
        canvas.oncontextmenu = function(e) {e.preventDefault()};
    }

    make_control_panel() {
        // make_control_panel(): Sets up a panel of interactive HTML elements, including
        // buttons with key bindings for affecting this scene, and live info readouts.
        this.control_panel.innerHTML += "Click and drag right mouse button to spin your viewpoint around it.<br>";
        this.live_string(box => box.textContent = "- Position: " + this.pos[0].toFixed(2) + ", " + this.pos[1].toFixed(2)
            + ", " + this.pos[2].toFixed(2));
        this.new_line();
        // The facing directions are surprisingly affected by the left hand rule:
        this.live_string(box => box.textContent = "- Facing: " + ((this.z_axis[0] > 0 ? "West " : "East ")
            + (this.z_axis[1] > 0 ? "Down " : "Up ") + (this.z_axis[2] > 0 ? "North" : "South")));
        this.new_line();
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

    display(context, graphics_state, dt = graphics_state.animation_delta_time / 1000) {
        // The whole process of acting upon controls begins here.
        const m = this.speed_multiplier * this.meters_per_frame,
            r = this.speed_multiplier * this.radians_per_frame;

        if (this.will_take_over_graphics_state) {
            this.reset(graphics_state);
            this.will_take_over_graphics_state = false;
        }

        if (!this.mouse_enabled_canvases.has(context.canvas)) {
            this.add_mouse_controls(context.canvas);
            this.mouse_enabled_canvases.add(context.canvas)
        }
        // Move in first-person.  Scale the normal camera aiming speed by dt for smoothness:
        this.first_person_flyaround(dt * r, dt * m);
        // Also apply third-person "arcball" camera mode if a mouse drag is occurring:
        if (this.mouse.anchor)
            this.third_person_arcball(dt * r);
        // Log some values:
        this.pos = this.inverse().times(vec4(0, 0, 0, 1));
        this.z_axis = this.inverse().times(vec4(0, 0, 1, 0));
    }
}

class Base_Scene extends Scene {
    constructor() {
        super();

        this.offsetsWidth = 256;
        this.offsetsLength = 256;
        this.offsets = [];
        for (let i = 0; i < this.offsetsWidth * this.offsetsLength; i++){
            this.offsets.push(0, 255, 255, 255);
        }

        this.texture = new Custom_Texture(this.offsetsLength, this.offsetsWidth, this.offsets);

        this.shapes = {
            'square' : new defs.Square(),
            'cube': new Cube(),
            'outline': new Cube_Outline(),
            'single_strip' : new Cube_Single_Strip(),
            'plane' : new Triangle_Strip_Plane(20,20, Vector3.create(0,0,0), 10),
            'axis' : new defs.Axis_Arrows()
        };

        this.materials = {
            plastic: new Material(new defs.Phong_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            offset: new Material(new Offset_shader(), {color: hex_color("#2b3b86"), texture: this.texture}),
            customOffset: new Material(new Offset_shader(), {color: hex_color("#2b3b86"), texture: this.texture})
        };
        this.white = new Material(new defs.Basic_Shader());
    }

    display(context, program_state) {
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new Custom_Movement_Controls(this.offsets, this.offsetsLength, this.offsetsWidth));
            program_state.set_camera(Mat4.look_at(vec3(6, 7, 25), vec3(0, 0, 0), vec3(0, 1, 0)))
        }
        program_state.projection_transform = Mat4.perspective(
            Math.PI / 4, context.width / context.height, 1, 100);

        const light_position = vec4(0, 5, 5, 1);
        program_state.lights = [new Light(light_position, color(1, 1, 1, 1), 1000)];
    }
}

export class Assignment2 extends Base_Scene {
    constructor() {
        super();
        this.clicked = 1;
    }


    make_control_panel() {
        this.key_triggered_button("Placeholder", ["c"], () => 1);
    }


    display(context, program_state) {
        super.display(context, program_state);

        for (let z = 0; z < this.offsetsWidth * 4; z+= 4){
            for (let x = 0; x < this.offsetsLength * 4; x+= 4){
                this.offsets[x+z*this.offsetsWidth] = 128 * (Math.sin(x/20 - program_state.animation_time/1000) + 1);
                this.offsets[x+(z*this.offsetsWidth)+1] = 0;
                this.offsets[x+(z*this.offsetsWidth)+2] = 255;
                this.offsets[x+(z*this.offsetsWidth)+3] = 255;
            }
        }
        this.texture.copy_onto_graphics_card(context.context, false);


        let model_transform = Mat4.identity();

        //this.shapes.plane.draw(context, program_state, model_transform, this.materials.offset, "TRIANGLE_STRIP");
        this.shapes.plane.draw(context, program_state, model_transform, this.materials.customOffset, "TRIANGLE_STRIP");
        this.shapes.axis.draw(context, program_state, model_transform, this.materials.plastic);
        //this.shapes.square.draw(context, program_state, model_transform, this.materials.customOffset);

        let intersect = this.shapes.plane.intersection(globOrig, globDir);
        let dest = intersect[0];
        dest[1] += this.offsets[intersect[1] * 4]/255;
        if (!dest){
            dest = Vector3.create(0,0,0);
        }
        model_transform = model_transform.times(Mat4.translation(dest[0], dest[1], dest[2]));
        this.shapes.axis.draw(context, program_state, model_transform, this.materials.plastic);
    }
}