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
        this.length = length;
        this.width = width;
        this.density = density;
        let denseWidth = width * density;
        let denseLength = length * density;
        //create vertex positions and texture coords. texture coords go from 0,1 in top left to 1,0 in bottom right and are
        // just normalized by percentage of the way from 0 -> number of wanted vertices
        for (let z = 0; z < denseWidth; z++){
            for (let x = 0; x < denseLength; x++){
                this.arrays.position.push(Vector3.create(x/density - length/2 + origin[0] + 1,origin[1],z/density - width/2 + origin[2] + 1));
                this.arrays.texture_coord.push(Vector.create(x/denseLength,1 - (z/denseWidth)));
                this.arrays.normal.push(Vector3.create(x/density - length/2 + origin[0] + 1,origin[1],z/density - width/2 + origin[2] + 1));
            }
        }

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
    closestVertexToRay(origin, direction){
        let minDistance = 999999999;
        let finalPos;

        //loop through each vertex in the shape and find the closest one. distanceVec is the distance between the origin and the vertex you are looking at
        //dest is the destination point made by moving the origin's location by distanceVec in the direction's direction
        //distance is the distance between this destination point and the vertex
        for (let i = 0; i < this.arrays.position.length; i++){
            let distanceNoDir = Vector3.create(origin[0], origin[1], origin[2]).minus(Vector3.create(this.arrays.position[i][0], 0, this.arrays.position[i][2])).norm();
            let dest = Vector3.create(direction[0], direction[1], direction[2]).times(distanceNoDir).plus(Vector3.create(origin[0], origin[1], origin[2]));
            let distance = Math.abs((dest.minus(Vector3.create(this.arrays.position[i][0], 0, this.arrays.position[i][2])).norm()));

            if (distance < minDistance){
                minDistance = distance;
                finalPos = Vector3.create(this.arrays.position[i][0], this.arrays.position[i][1], this.arrays.position[i][2]);
            }
        }
        return finalPos;
    }

    updateVertexHeight(x, z, newHeight){
        if ((x + (z * this.width * this.density)) < this.arrays.position.length && (x + (z * this.width * this.density)) >= 0) {
            this.arrays.position[x + (z * this.width * this.density)][1] = newHeight;
        }
    }

    addVertexHeight(x, z, newHeight){
        if ((x + (z * this.width * this.density)) < this.arrays.position.length && (x + (z * this.width * this.density)) >= 0) {
            this.arrays.position[x + (z * this.width * this.density)][1] -= newHeight;
        }
    }
}

// custom shader class that takes in a texture and uses that texture's red value to offset the vertex's y position
// paints the texture onto the shape as color. Eventually I think maybe we can pass in 2 textures, one as a heightmap
// and one as the actual texture we want to be displayed. To learn how these work check out the examples in common.js
class Offset_shader extends Shader {
    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
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
                    //gl_Position = projection_camera_model_transform * vec4( position.x, position.y + tex_color.r, position.z, 1.0 );
                    gl_Position = projection_camera_model_transform * vec4( position.x, position.y, position.z, 1.0 );
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

class Phong_Water_Shader extends Shader{

    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000);
        context.uniform4fv(gpu_addresses.shape_color, material.color);
        context.uniform1f(gpu_addresses.ambient, material.ambient);
        context.uniform1f(gpu_addresses.diffusivity, material.diffusivity);
        context.uniform1f(gpu_addresses.specularity, material.specularity);
        context.uniform1f(gpu_addresses.smoothness, material.smoothness);
        const O = vec4(0, 0, 0, 1), camera_center = graphics_state.camera_transform.times(O).to3();
        context.uniform3fv(gpu_addresses.camera_center, camera_center);
        const squared_scale = model_transform.reduce(
            (acc, r) => {
                return acc.plus(vec4(...r).times_pairwise(r))
            }, vec4(0, 0, 0, 0)).to3();
        context.uniform3fv(gpu_addresses.squared_scale, squared_scale);

        if (!graphics_state.lights.length)
            return;

        const light_positions_flattened = [], light_colors_flattened = [];
        for (let i = 0; i < 4 * graphics_state.lights.length; i++) {
            light_positions_flattened.push(graphics_state.lights[Math.floor(i / 4)].position[i % 4]);
            light_colors_flattened.push(graphics_state.lights[Math.floor(i / 4)].color[i % 4]);
        }
        context.uniform4fv(gpu_addresses.light_positions_or_vectors, light_positions_flattened);
        context.uniform4fv(gpu_addresses.light_colors, light_colors_flattened);
        context.uniform1fv(gpu_addresses.light_attenuation_factors, graphics_state.lights.map(l => l.attenuation));
    }

    shared_glsl_code() {
        return `precision mediump float;
                varying vec2 noisePos;
                uniform float time;
                
                vec2 voronoiSRand(vec2 value){
                  return vec2(sin(fract(sin(dot(sin(value), vec2(12.989, 78.233))) * 143758.5453) * (time + 150.0)) * 0.5 + 0.5, 
                      cos(fract(sin(dot(sin(value), vec2(39.346, 11.135))) * 143758.5453) * (time + 13.0)) * 0.5 + 0.5);
                }
                
                float voronoi(vec2 pos){
                  vec2 baseCell = floor(pos);
                  float minDist = 10.0;
                  
                  for (int i = -1; i <= 1; i++){
                    for (int j = -1; j <= 1; j++){
                      vec2 currentCell = baseCell + vec2(i, j);
                      vec2 posInCell = currentCell + voronoiSRand(currentCell);
                      minDist = min(distance(posInCell, pos), minDist);
                    }
                  }
                  return minDist;
                }
                
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec3 squared_scale, camera_center;
                uniform vec4 shape_color;
        
                varying vec3 N, vertex_worldspace;
                vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace ){                                        
                    vec3 E = normalize( camera_center - vertex_worldspace );
                    vec3 result = vec3( 0.0 );
                    for(int i = 0; i < N_LIGHTS; i++){
                        vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                                       light_positions_or_vectors[i].w * vertex_worldspace;                                             
                        float distance_to_light = length( surface_to_light_vector );
        
                        vec3 L = normalize( surface_to_light_vector );
                        vec3 H = normalize( L + E );

                        float diffuse  =      max( dot( N, L ), 0.0 );
                        float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                        float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light);
                        
                        vec3 light_contribution = shape_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                        result += attenuation * light_contribution;
                      }
                    return result;
                  }
            `;
    }

    //sets each vertex's position to the its position + the red value of our texture
    vertex_glsl_code() {
        return this.shared_glsl_code() + `
                attribute vec3 position;
                attribute vec3 normal;                         
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                
                void main(){
                    noisePos = (model_transform * vec4(position.x, position.y, position.z, 1.0)).xz * (2.0 / 1.0);
                    gl_Position = projection_camera_model_transform * vec4( position.x, position.y + (voronoi(noisePos) / 10.0), position.z, 1.0 );
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                }`;
    }

    //sets each pixel's color
    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                
                void main(){
                    gl_FragColor = vec4(shape_color.x * ambient, shape_color.y * ambient, shape_color.z * ambient, 0.95);
                    vec3 lighting = phong_model_lights( normalize( N ), vertex_worldspace);
                    gl_FragColor.xyz += lighting + (pow(voronoi(noisePos), 3.0)) * lighting * 3.0 + (pow(voronoi(noisePos), 3.0)) * (ambient / 3.0);
                }`;
    }
}

class Water_Shader extends Shader{

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000);
        context.uniform4fv(gpu_addresses.shape_color, material.color);
    }

    shared_glsl_code() {
        return `precision mediump float;
                varying vec2 noisePos;
                uniform float time;
                uniform vec4 shape_color;
                
                vec2 voronoiSRand(vec2 value){
                  return vec2(sin(fract(sin(dot(sin(value), vec2(12.989, 78.233))) * 143758.5453) * (time + 150.0)) * 0.5 + 0.5, 
                      cos(fract(sin(dot(sin(value), vec2(39.346, 11.135))) * 143758.5453) * (time + 13.0)) * 0.5 + 0.5);
                }
                
                float voronoi(vec2 pos){
                  vec2 baseCell = floor(pos);
                  float minDist = 10.0;
                  
                  for (int i = -1; i <= 1; i++){
                    for (int j = -1; j <= 1; j++){
                      vec2 currentCell = baseCell + vec2(i, j);
                      vec2 posInCell = currentCell + voronoiSRand(currentCell);
                      minDist = min(distance(posInCell, pos), minDist);
                    }
                  }
                  return minDist;
                }
            `;
    }

    //sets each vertex's position to the its position + the red value of our texture
    vertex_glsl_code() {
        return this.shared_glsl_code() + `
                attribute vec3 position;
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                
                void main(){
                    noisePos = (model_transform * vec4(position.x, position.y, position.z, 1.0)).xz * (2.0 / 1.0);
                    gl_Position = projection_camera_model_transform * vec4( position.x, position.y + (voronoi(noisePos) / 10.0), position.z, 1.0 );
                }`;
    }

    //sets each pixel's color
    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                void main(){
                    gl_FragColor = vec4(shape_color.xyz + (vec3(0.0,0.8, 0.6) * pow(voronoi(noisePos), 3.0)), 0.95);
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

class Scene_Object{
    constructor(shape, transform, material, renderArgs = "TRIANGLES") {
        this.shape = shape;
        this.transform = transform;
        this.material = material;
        this.renderArgs = renderArgs;
    }

    drawObject(context, program_state){
        this.shape.draw(context, program_state, this.transform, this.material, this.renderArgs);
    }
}

class Base_Scene extends Scene {
    constructor() {
        super();

        //define how long and wide we want our plane to be. the density gives extra resolution by making more triangles in the same amount of space
        this.planeWidth = 20;
        this.planeLength = 20;
        this.planeDensity = 7;

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
            'plane' : new Triangle_Strip_Plane(this.planeLength,this.planeWidth, Vector3.create(0,0,0), this.planeDensity),
            'axis' : new defs.Axis_Arrows()
        };

        this.materials = {
            plastic: new Material(new defs.Phong_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            texturedPhong: new Material(new defs.Textured_Phong(),
                {ambient: 0.4, diffusivity: 0.6, specularity: 0.5, color: hex_color("#494949"), texture: new Texture("assets/rgb.jpg")}),
            bump: new Material(new defs.Textured_Phong(),
                {ambient: .4, diffusivity: .6, color: hex_color("#ffffff"), texture: new Texture("assets/rgb.jpg")}),
            //creates a material based on our texture. For now it also takes a color variable, but I think we can get rid of that at some point
            offset: new Material(new Offset_shader(), {color: hex_color("#2b3b86"), texture: this.texture}),
            phong_water: new Material(new Phong_Water_Shader(), {color: hex_color("#4e6ef6"), ambient: 0.2, diffusivity: 0.7, specularity: 0.03, smoothness: 100}),
            funny: new Material(new defs.Funny_Shader()),
            water: new Material(new Water_Shader(), {color: hex_color("#4e6ef6")}),
        };
        this.white = new Material(new defs.Basic_Shader());

        this.water_plane = new Scene_Object(new Triangle_Strip_Plane(15,15, Vector3.create(0,0,0), this.planeDensity), Mat4.translation(0,-0.7,0), this.materials.phong_water, "TRIANGLE_STRIP");
        this.plane = new Scene_Object(new Triangle_Strip_Plane(this.planeLength,this.planeWidth, Vector3.create(0,0,0), this.planeDensity), Mat4.translation(0,0,0), this.materials.plastic, "TRIANGLE_STRIP");
    }

    display(context, program_state) {
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new Custom_Movement_Controls());
            //sets the camera at location (6,7,25), looking at the origin
            program_state.set_camera(Mat4.look_at(vec3(6, 7, 25), vec3(0, 0, 0), vec3(0, 1, 0)))
        }
        program_state.projection_transform = Mat4.perspective(
            Math.PI/4 , context.width / context.height, 1, 100);

        program_state.lights = [new Light(vec4(7, 2, 7, 0), color(1.1, 1.1, 1.1, 1), 10000), new Light(vec4(-7, 2, -7, 0), color(0, 1, 0, 1), 10000)];
    }
}

export class Test extends Base_Scene {
    constructor() {
        super();
    }

    make_control_panel(context) {
        this.key_triggered_button("change water shader", ["p"], () => this.water_plane.material = (this.water_plane.material === this.materials.water) ? this.materials.phong_water:this.materials.water);

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
        //this is the world space coordinates of the near point. We divide by the last homogenous value because of the perspective transform
        worldSpaceNear = Vector.create(worldSpaceNear[0] / worldSpaceNear[3], worldSpaceNear[1] / worldSpaceNear[3], worldSpaceNear[2] / worldSpaceNear[3], 1);

        let mousePointFar = Vector.create(mousePosPercent[0], mousePosPercent[1], 1, 1);
        let worldSpaceFar = transformMatrix.times(mousePointFar);
        //this is the world space coordinates of the far point. We divide by the last homogenous value because of the perspective transform
        worldSpaceFar = Vector.create(worldSpaceFar[0] / worldSpaceFar[3], worldSpaceFar[1] / worldSpaceFar[3], worldSpaceFar[2] / worldSpaceFar[3], 1);

        //this calls the intersection function of the plane shape to find which vertex is nearest to the mouse click. the function takes
        //our mouse's location as the origin, and a vector direction to the world space location of the far coordinate
        return this.plane.shape.closestVertexToRay(worldSpaceNear, worldSpaceFar.minus(worldSpaceNear).normalized());
    }

    //this function draws onto the heightmap texture when given a location to draw (in world space coordinates) and a brush radius.
    //eventually I intend for this to create a circle of that radius, but for now its just a square of size 10
    drawnOnTexture(location, brushRadius) {
        //this find what percentage to the edge the location given is. I currently treat the texture as taking up the same world space size as the size of the plane
        let textureLocPercent = Vector.create((location[0]-1) / (this.planeWidth / 2), (location[2]-1) / (this.planeLength / 2));
        //using the percentage we can find what z and x coordinates that would be given 128 rows to the right and down from the origin
        let textureLoc = Vector.create(Math.ceil(textureLocPercent[0] * 128) + 128, Math.ceil(textureLocPercent[1] * 128) + 128);

        let strength = 20;
        for (let i = 0; i < brushRadius; i++) {
            for (let dy = 0; dy < i; dy++) {
                let dx = 0;
                let sq = (i * i) - (dy * dy);
                while ((dx * dx) < sq) {
                    this.offsets[((dx + textureLoc[0]) * 4) + ((dy + textureLoc[1]) * 4 * 256)] += strength;
                    this.offsets[((dx + textureLoc[0]) * 4) + ((-dy + textureLoc[1] - 1) * 4 * 256)] += strength;
                    this.offsets[((-dx + textureLoc[0] - 1) * 4) + ((dy + textureLoc[1]) * 4 * 256)] += strength;
                    this.offsets[((-dx + textureLoc[0] - 1) * 4) + ((-dy + textureLoc[1] - 1) * 4 * 256)] += strength;
                    dx++;
                }
            }
        }
    }

    drawnOnPlane(location, brushRadius) {
        //this find what percentage to the edge the location given is. I currently treat the texture as taking up the same world space size as the size of the plane
        let planeLocPercent = Vector.create((location[0]-1) / (this.planeWidth / 2), (location[2]-1) / (this.planeLength / 2));
        //using the percentage we can find what z and x coordinates that would be given 128 rows to the right and down from the origin
        let planeLoc = Vector.create(Math.ceil(planeLocPercent[0] * (this.planeWidth * this.planeDensity / 2)) + (this.planeWidth * this.planeDensity / 2), Math.ceil(planeLocPercent[1] * (this.planeWidth * this.planeDensity / 2)) + (this.planeWidth * this.planeDensity / 2));

        let strength = 0.05 / Math.max((brushRadius - 6), 1);
        for (let i = 0; i < brushRadius; i++) {
            for (let dy = 0; dy < i; dy++) {
                let dx = 0;
                let sq = (i * i) - (dy * dy);
                while ((dx * dx) < sq) {
                    this.plane.shape.addVertexHeight(dx + planeLoc[0], dy + planeLoc[1], strength);
                    this.plane.shape.addVertexHeight(dx + planeLoc[0], -dy + planeLoc[1] - 1, strength);
                    this.plane.shape.addVertexHeight(-dx + planeLoc[0] - 1, dy + planeLoc[1], strength);
                    this.plane.shape.addVertexHeight(-dx + planeLoc[0] - 1, -dy + planeLoc[1] - 1, strength);
                    dx++;
                }
            }
        }
    }

    display(context, program_state) {
        super.display(context, program_state);

        //if the mouse is held down and we are trying to paint on the canvas
        if (context.scratchpad.controls.mouse.isPainting) {
            //find the location we are trying to paint at
            let dest = this.getClosestLocOnPlane(context, program_state);
            //paint on the texture with a brush radius of 10 (the brush radius doesn't actually do anything yet)
            this.drawnOnTexture(dest, 25);
            this.texture.copy_onto_graphics_card(context.context, false);
            this.drawnOnPlane(dest, 20);
            this.plane.shape.copy_onto_graphics_card(context.context);
        }

        //draw the plane and axis (axis was just so I could see if it is actually centered, I should honestly just remove it)
        let model_transform = Mat4.identity();
        this.shapes.axis.draw(context, program_state, model_transform, this.materials.plastic);
        this.plane.drawObject(context, program_state);
        this.water_plane.drawObject(context, program_state);
    }
}