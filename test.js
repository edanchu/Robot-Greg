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
// (eg: density 10 means that you'll get 10 vertices between (0,0,0) and (1,0,0))
class Triangle_Strip_Plane extends Shape{
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

//im not gonna comment on how the shaders work for now. If you want to know how they work, ask me in person and ill go through them
//This shader creates the skybox by blending between 3 values, one for the horizon, under the horizon, and above the horizon
class Skybox_Shader extends Shader {

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniform4fv(gpu_addresses.top_color, material.top_color);
        context.uniform4fv(gpu_addresses.mid_color, material.mid_color);
        context.uniform4fv(gpu_addresses.bottom_color, material.bottom_color);
    }
    shared_glsl_code() {
        return `precision mediump float;
               varying vec4 pos;
           `;
    }

    vertex_glsl_code() {
        return this.shared_glsl_code() + `
               attribute vec3 position;    
               uniform mat4 projection_camera_model_transform;
        
                void main(){
                    gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                    pos = vec4(position, 1.0);
                }`;
    }

    fragment_glsl_code() {
        return this.shared_glsl_code() + `
               uniform vec4 top_color;
               uniform vec4 mid_color;
               uniform vec4 bottom_color;
                
                void main(){
                    float topGrad = pow(max(0.0, pos.y), 0.75);
                    float bottomGrad = pow(min(0.0, pos.y) * -1.0, 0.75);
                    vec4 midColor = (1.0 - (topGrad + bottomGrad)) * mid_color;
                    vec4 topColor = topGrad * top_color;
                    vec4 bottomColor = bottomGrad * bottom_color;
                    gl_FragColor = topColor + bottomColor + midColor;
                }`;
    }
}

class PlainShader extends Shader {

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
    }
    shared_glsl_code() {
        return `precision mediump float;
           `;
    }

    vertex_glsl_code() {
        return this.shared_glsl_code() + `
               attribute vec3 position;    
               uniform mat4 projection_camera_model_transform;
        
                void main(){
                    gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                }`;
    }

    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                void main(){
                    gl_FragColor = vec4(1.0,1.0,1.0,1.0);
                }`;
    }
}

//the grass shader for the terrain. Has built in Phong lighting. To work correctly it needs to have its layer property set
//and incremented in successive draw calls. I find that drawing with 16 layers looks good enough
//doesn't draw grass when the vertex's world position is under a certain threshold (so no grass under water)
//no grass will be drawn where the occlusion texture's red value is not 0
class Grass_Shader extends Shader {
    constructor(layer, num_lights = 2) {
        super();
        this.layer = layer;
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniform4fv(gpu_addresses.color, material.color);
        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000.0);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniform1f(gpu_addresses.layer, this.layer);
        context.uniform1i(gpu_addresses.texture, 0);
        material.texture.activate(context);

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
        
                varying vec4 worldPos;
                uniform float time;
                uniform float layer;
                
                uniform sampler2D texture;
                varying vec2 f_tex_coord;
                
                float random (vec2 value){
                    return fract(sin(dot(value, vec2(94.8365, 47.053))) * 94762.9342);
                }

                float lerp(float a, float b, float percent){
                    return (1.0 - percent) * a + (percent * b);
                }

                float perlinNoise (vec2 value){
                   vec2 integer = floor(value);
                   vec2 fractional = fract(value);
                   fractional = fractional * fractional * (3.0 - 2.0 * fractional);

                   value = abs(fract(value) - 0.5);
                   float currCell = random(integer + vec2(0.0, 0.0));
                   float rightCell = random(integer + vec2(1.0, 0.0));
                   float bottomCell = random(integer + vec2(0.0, 1.0));
                   float bottomRightCell = random(integer + vec2(1.0, 1.0));

                   float currRow = lerp(currCell, rightCell, fractional.x);
                   float lowerRow = lerp(bottomCell, bottomRightCell, fractional.x);
                   float lerpedRandomVal = lerp(currRow, lowerRow, fractional.y);
                   return lerpedRandomVal;
                }

                float PerlinNoise3Pass(vec2 value, float Scale){
                    float outVal = 0.0;

                    float frequency = pow(2.0, 0.0);
                    float amplitude = pow(0.5, 3.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    frequency = pow(2.0, 1.0);
                    amplitude = pow(0.5, 2.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    frequency = pow(2.0, 2.0);
                    amplitude = pow(0.5, 1.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    return outVal;
                }
                
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec3 squared_scale, camera_center;
                uniform vec4 color;
        
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
                        
                        vec3 light_contribution = color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                        result += attenuation * light_contribution;
                      }
                    return result;
                  }
        
            `;
    }

    vertex_glsl_code() {
        return this.shared_glsl_code() + `
                attribute vec3 position;
                attribute vec2 texture_coord;  
                attribute vec3 normal;                       
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                
                void main(){
                    worldPos = model_transform * vec4(position, 1.0);
                    float alpha = (layer / 10.0) * PerlinNoise3Pass(worldPos.xz + vec2(time * 2.0, time * 2.0), 2.0);
                    gl_Position = projection_camera_model_transform * vec4(position.x + (0.2 * alpha), position.y + (0.04 * layer), position.z + (0.2 * alpha), 1.0);
                    f_tex_coord = texture_coord;
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                }`;
    }

    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                void main(){
                    gl_FragColor = vec4(color.x * ambient + (layer / 70.0), color.y * ambient + (layer / 70.0), color.z * ambient + (layer / 70.0), 1.0);
                    gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace);
                    
                    if (layer > 0.0){
                        float perlin = 1.0 - (1.0 - PerlinNoise3Pass(worldPos.xz, 50.0)) * 2.2;
                        float white = 1.0 - (1.0 - perlinNoise(worldPos.xz)) * 40.0;
                        float alpha = perlin * white - ((layer + 0.2) * 1.2 / 1.0);
                        if (alpha < 0.0 || worldPos.y < -0.7){
                            discard;
                        }
                        vec4 tex_color = texture2D(texture, f_tex_coord);
                        if (tex_color.r > 0.0){
                            discard;
                        }
                    }
                }`;
    }
}

class Grass_Shader_Shadow extends Shader {
    constructor(layer, num_lights = 2) {
        super();
        this.layer = layer;
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniform4fv(gpu_addresses.ground_color, material.ground_color);
        context.uniform4fv(gpu_addresses.grass_color, material.grass_color);

        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000.0);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniform1f(gpu_addresses.layer, this.layer);
        context.uniform1i(gpu_addresses.texture, 0);
        material.texture.activate(context);
        context.uniform1i(gpu_addresses.fake_shadow_layer, material.fake_shadow_layer);
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

        context.uniformMatrix4fv(gpu_addresses.light_view_mat, false, Matrix.flatten_2D_to_1D(material.light_view_mat.transposed()));
        context.uniformMatrix4fv(gpu_addresses.light_proj_mat, false, Matrix.flatten_2D_to_1D(material.light_proj_mat.transposed()));
        context.uniform1i(gpu_addresses.draw_shadow, material.draw_shadow);
        context.uniform1f(gpu_addresses.light_depth_bias, 0.003);
        context.uniform1f(gpu_addresses.light_texture_size, material.lightDepthTextureSize);
        context.uniform1i(gpu_addresses.light_depth_texture, 1);
        if (material.draw_shadow) {
            material.light_depth_texture.activate(context, 1);
        }
    }

    shared_glsl_code() {
        return `precision mediump float;
        
                varying vec4 worldPos;
                uniform float time;
                uniform float layer;
                
                uniform sampler2D texture;
                varying vec2 f_tex_coord;
                
                float random (vec2 value){
                    return fract(sin(dot(value, vec2(94.8365, 47.053))) * 94762.9342);
                }

                float lerp(float a, float b, float percent){
                    return (1.0 - percent) * a + (percent * b);
                }

                float perlinNoise (vec2 value){
                   vec2 integer = floor(value);
                   vec2 fractional = fract(value);
                   fractional = fractional * fractional * (3.0 - 2.0 * fractional);

                   value = abs(fract(value) - 0.5);
                   float currCell = random(integer + vec2(0.0, 0.0));
                   float rightCell = random(integer + vec2(1.0, 0.0));
                   float bottomCell = random(integer + vec2(0.0, 1.0));
                   float bottomRightCell = random(integer + vec2(1.0, 1.0));

                   float currRow = lerp(currCell, rightCell, fractional.x);
                   float lowerRow = lerp(bottomCell, bottomRightCell, fractional.x);
                   float lerpedRandomVal = lerp(currRow, lowerRow, fractional.y);
                   return lerpedRandomVal;
                }

                float PerlinNoise3Pass(vec2 value, float Scale){
                    float outVal = 0.0;

                    float frequency = pow(2.0, 0.0);
                    float amplitude = pow(0.5, 3.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    frequency = pow(2.0, 1.0);
                    amplitude = pow(0.5, 2.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    frequency = pow(2.0, 2.0);
                    amplitude = pow(0.5, 1.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    return outVal;
                }
                
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec3 squared_scale, camera_center;
                uniform vec4 ground_color;
                uniform vec4 grass_color;
        
                varying vec3 N, vertex_worldspace;
                vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace, 
                        out vec3 light_diffuse_contribution, out vec3 light_specular_contribution ){                                        
                    // phong_model_lights():  Add up the lights' contributions.
                    vec3 E = normalize( camera_center - vertex_worldspace );
                    vec3 result = vec3( 0.0 );
                    light_diffuse_contribution = vec3( 0.0 );
                    light_specular_contribution = vec3( 0.0 );
                    for(int i = 0; i < N_LIGHTS; i++){
                        // Lights store homogeneous coords - either a position or vector.  If w is 0, the 
                        // light will appear directional (uniform direction from all points), and we 
                        // simply obtain a vector towards the light by directly using the stored value.
                        // Otherwise if w is 1 it will appear as a point light -- compute the vector to 
                        // the point light's location from the current surface point.  In either case, 
                        // fade (attenuate) the light as the vector needed to reach it gets longer.  
                        vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                                       light_positions_or_vectors[i].w * vertex_worldspace;                                             
                        float distance_to_light = length( surface_to_light_vector );
        
                        vec3 L = normalize( surface_to_light_vector );
                        vec3 H = normalize( L + E );
                        // Compute the diffuse and specular components from the Phong
                        // Reflection Model, using Blinn's "halfway vector" method:
                        float diffuse  =      max( dot( N, L ), 0.0 );
                        float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                        float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                        
                        vec3 light_contribution = grass_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                        light_diffuse_contribution += attenuation * grass_color.xyz * light_colors[i].xyz * diffusivity * diffuse;
                        light_specular_contribution += attenuation * grass_color.xyz * specularity * specular;
                        result += attenuation * light_contribution;
                      }
                    return result;
                  }
        
            `;
    }

    vertex_glsl_code() {
        return this.shared_glsl_code() + `
                attribute vec3 position;
                attribute vec2 texture_coord;  
                attribute vec3 normal;                       
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                
                void main(){
                    worldPos = model_transform * vec4(position, 1.0);
                    float alpha = (layer / 10.0) * PerlinNoise3Pass(worldPos.xz + vec2(time * 2.0, time * 2.0), 2.0);
                    gl_Position = projection_camera_model_transform * vec4(position.x + (0.2 * alpha), position.y + (0.04 * layer), position.z + (0.2 * alpha), 1.0);
                    f_tex_coord = texture_coord;
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                }`;
    }

    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                uniform sampler2D light_depth_texture;
                uniform float light_texture_size;
                uniform mat4 light_view_mat;
                uniform mat4 light_proj_mat;
                uniform float light_depth_bias;
                uniform bool draw_shadow;
                uniform bool fake_shadow_layer;
                
                float PCF_shadow(vec2 center, float projected_depth) {
                    float shadow = 0.0;
                    float texel_size = 1.0 / light_texture_size;
                    for(int x = -1; x <= 1; ++x)
                    {
                        for(int y = -1; y <= 1; ++y)
                        {
                            float light_depth_value = texture2D(light_depth_texture, center + vec2(x, y) * texel_size).r; 
                            shadow += (projected_depth >= light_depth_value + light_depth_bias) ? 0.8 : 0.0;        
                        }    
                    }
                    shadow /= 9.0;
                    return shadow;
                }
        
                void main(){
                    if (fake_shadow_layer){
                        float perlin = 1.0 - (1.0 - PerlinNoise3Pass(vec2(worldPos.x + 0.03, worldPos.z - 0.03), 50.0)) * 2.2;
                        float white = 1.0 - (1.0 - perlinNoise(vec2(worldPos.x + 0.03, worldPos.z - 0.03))) * 40.0;
                        float alpha = perlin * white - ((layer + 0.2) * 1.8 / 1.0);
                        if (alpha < 0.0 || worldPos.y < -0.7){
                            discard;
                        }
                        gl_FragColor = vec4(grass_color.x * ambient * 1.3 + (layer / 200.0), grass_color.y * ambient * 1.3 + (layer / 200.0), grass_color.z * ambient * 1.3 + (layer / 200.0), 1.0);
                        vec4 tex_color = texture2D(texture, f_tex_coord);
                        if (tex_color.r > 0.0){
                            discard;
                        }
                    }
                    else {
                    if (layer > 0.0){
                        gl_FragColor = vec4(grass_color.x * ambient + (layer / 70.0), grass_color.y * ambient + (layer / 70.0), grass_color.z * ambient + (layer / 70.0), 1.0);
                    }
                    else {
                        gl_FragColor = vec4(ground_color.x * ambient, ground_color.y * ambient, ground_color.z * ambient, 1.0);
                    }
                    vec3 diffuse, specular;
                    vec3 other_than_ambient = phong_model_lights( normalize( N ), vertex_worldspace, diffuse, specular );
                    
                    if (draw_shadow) {
                        vec4 light_tex_coord = (light_proj_mat * light_view_mat * vec4(vertex_worldspace, 1.0));
                        light_tex_coord.xyz /= light_tex_coord.w; 
                        light_tex_coord.xyz *= 0.5;
                        light_tex_coord.xyz += 0.5;
                        float light_depth_value = texture2D( light_depth_texture, light_tex_coord.xy ).r;
                        float projected_depth = light_tex_coord.z;
                        
                        bool inRange =
                            light_tex_coord.x >= 0.0 &&
                            light_tex_coord.x <= 1.0 &&
                            light_tex_coord.y >= 0.0 &&
                            light_tex_coord.y <= 1.0;
                              
                        float shadowness = PCF_shadow(light_tex_coord.xy, projected_depth);
                        
                        if (inRange && shadowness > 0.3) {
                            diffuse *= 0.2 + 0.8 * (1.0 - shadowness);
                            specular *= 1.0 - shadowness;
                        }
                    }
                    
                    gl_FragColor.xyz += diffuse + specular;
                    
                    if (layer > 0.0){
                        float perlin = 1.0 - (1.0 - PerlinNoise3Pass(worldPos.xz, 50.0)) * 2.2;
                        float white = 1.0 - (1.0 - perlinNoise(worldPos.xz)) * 40.0;
                        float alpha = perlin * white - ((layer + 0.2) * 1.2 / 1.0);
                        if (alpha < 0.0 || worldPos.y < -0.7){
                            discard;
                        }
                        vec4 tex_color = texture2D(texture, f_tex_coord);
                        if (tex_color.r > 0.0){
                            discard;
                        }
                    }
                    }
                }`;
    }
}

//similar to the shader above, but doesn't draw the plane in the center area and doesn't have the checks for height or occlusion tex.
//Also, it has a easing function alpha fading far from the world origin for a fog effect.
class Grass_Shader_Background extends Shader {
    constructor(layer, num_lights = 2) {
        super();
        this.layer = layer;
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniform4fv(gpu_addresses.ground_color, material.ground_color);
        context.uniform4fv(gpu_addresses.grass_color, material.grass_color);
        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000.0);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniform1f(gpu_addresses.layer, this.layer);

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
        
                varying vec4 worldPos;
                uniform float time;
                uniform float layer;
                                
                float random (vec2 value){
                    return fract(sin(dot(value, vec2(94.8365, 47.053))) * 94762.9342);
                }

                float lerp(float a, float b, float percent){
                    return (1.0 - percent) * a + (percent * b);
                }

                float perlinNoise (vec2 value){
                   vec2 integer = floor(value);
                   vec2 fractional = fract(value);
                   fractional = fractional * fractional * (3.0 - 2.0 * fractional);

                   value = abs(fract(value) - 0.5);
                   float currCell = random(integer + vec2(0.0, 0.0));
                   float rightCell = random(integer + vec2(1.0, 0.0));
                   float bottomCell = random(integer + vec2(0.0, 1.0));
                   float bottomRightCell = random(integer + vec2(1.0, 1.0));

                   float currRow = lerp(currCell, rightCell, fractional.x);
                   float lowerRow = lerp(bottomCell, bottomRightCell, fractional.x);
                   float lerpedRandomVal = lerp(currRow, lowerRow, fractional.y);
                   return lerpedRandomVal;
                }

                float PerlinNoise3Pass(vec2 value, float Scale){
                    float outVal = 0.0;

                    float frequency = pow(2.0, 0.0);
                    float amplitude = pow(0.5, 3.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    frequency = pow(2.0, 1.0);
                    amplitude = pow(0.5, 2.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    frequency = pow(2.0, 2.0);
                    amplitude = pow(0.5, 1.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    return outVal;
                }
                
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec3 squared_scale, camera_center;
                uniform vec4 ground_color;
                uniform vec4 grass_color;
        
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
                        
                        vec3 light_contribution = grass_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                        result += attenuation * light_contribution;
                      }
                    return result;
                  }
        
            `;
    }

    vertex_glsl_code() {
        return this.shared_glsl_code() + `
                attribute vec3 position;
                attribute vec3 normal;                       
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                
                void main(){
                    worldPos = model_transform * vec4(position, 1.0);
                    float alpha = (layer / 10.0) * PerlinNoise3Pass(worldPos.xz + random(worldPos.xz) + vec2(time * 1.0, time * 1.0), 2.0) - 1.0;
                    gl_Position = projection_camera_model_transform * vec4(position.x + (0.02 * alpha), position.y + (0.04 * layer), position.z + (0.02 * alpha), 1.0);
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                }`;
    }

    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                void main(){
                    if (layer > 0.0){
                        gl_FragColor = vec4(grass_color.x * ambient + (layer / 70.0), grass_color.y * ambient + (layer / 70.0), grass_color.z * ambient + (layer / 70.0), 1.0 - exp(-0.5 * (40.0 - distance(vec4(0,0,0,0), worldPos))));
                    }
                    else {
                        gl_FragColor = vec4(ground_color.x * ambient, ground_color.y * ambient, ground_color.z * ambient, 1.0 - exp(-0.5 * (40.0 - distance(vec4(0,0,0,0), worldPos))));
                    }
                    gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace);
                    
                    if ((worldPos.x < 13.5 && worldPos.x > -11.5) && (worldPos.z < 13.5 && worldPos.z > -11.5)){
                        discard;
                    }
                    if (layer > 0.0){
                        float perlin = 1.0 - (1.0 - PerlinNoise3Pass(worldPos.xz, 55.0)) * 2.2;
                        float white = 1.0 - (1.0 - perlinNoise(worldPos.xz)) * 40.0;
                        float alpha = perlin * white - ((layer + 0.2) * 1.2 / 1.0);
                        if (alpha < 0.0){
                            discard;
                        }
                    }
                }`;
    }
}

class Grass_Shader_Shadow_Background extends Shader {
    constructor(layer, num_lights = 2) {
        super();
        this.layer = layer;
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniform4fv(gpu_addresses.ground_color, material.ground_color);
        context.uniform4fv(gpu_addresses.grass_color, material.grass_color);

        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000.0);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniform1f(gpu_addresses.layer, this.layer);
        context.uniform1i(gpu_addresses.texture, 0);
        material.texture.activate(context);
        context.uniform1i(gpu_addresses.fake_shadow_layer, material.fake_shadow_layer);
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

        context.uniformMatrix4fv(gpu_addresses.light_view_mat, false, Matrix.flatten_2D_to_1D(material.light_view_mat.transposed()));
        context.uniformMatrix4fv(gpu_addresses.light_proj_mat, false, Matrix.flatten_2D_to_1D(material.light_proj_mat.transposed()));
        context.uniform1i(gpu_addresses.draw_shadow, material.draw_shadow);
        context.uniform1f(gpu_addresses.light_depth_bias, 0.003);
        context.uniform1f(gpu_addresses.light_texture_size, material.lightDepthTextureSize);
        context.uniform1i(gpu_addresses.light_depth_texture, 1);
        if (material.draw_shadow) {
            material.light_depth_texture.activate(context, 1);
        }
    }

    shared_glsl_code() {
        return `precision mediump float;
        
                varying vec4 worldPos;
                uniform float time;
                uniform float layer;
                
                uniform sampler2D texture;
                varying vec2 f_tex_coord;
                
                float random (vec2 value){
                    return fract(sin(dot(value, vec2(94.8365, 47.053))) * 94762.9342);
                }

                float lerp(float a, float b, float percent){
                    return (1.0 - percent) * a + (percent * b);
                }

                float perlinNoise (vec2 value){
                   vec2 integer = floor(value);
                   vec2 fractional = fract(value);
                   fractional = fractional * fractional * (3.0 - 2.0 * fractional);

                   value = abs(fract(value) - 0.5);
                   float currCell = random(integer + vec2(0.0, 0.0));
                   float rightCell = random(integer + vec2(1.0, 0.0));
                   float bottomCell = random(integer + vec2(0.0, 1.0));
                   float bottomRightCell = random(integer + vec2(1.0, 1.0));

                   float currRow = lerp(currCell, rightCell, fractional.x);
                   float lowerRow = lerp(bottomCell, bottomRightCell, fractional.x);
                   float lerpedRandomVal = lerp(currRow, lowerRow, fractional.y);
                   return lerpedRandomVal;
                }

                float PerlinNoise3Pass(vec2 value, float Scale){
                    float outVal = 0.0;

                    float frequency = pow(2.0, 0.0);
                    float amplitude = pow(0.5, 3.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    frequency = pow(2.0, 1.0);
                    amplitude = pow(0.5, 2.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    frequency = pow(2.0, 2.0);
                    amplitude = pow(0.5, 1.0);
                    outVal += perlinNoise(vec2(value.x * Scale / frequency, value.y * Scale / frequency)) * amplitude;

                    return outVal;
                }
                
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec3 squared_scale, camera_center;
                uniform vec4 ground_color;
                uniform vec4 grass_color;
        
                varying vec3 N, vertex_worldspace;
                vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace, 
                        out vec3 light_diffuse_contribution, out vec3 light_specular_contribution ){                                        
                    // phong_model_lights():  Add up the lights' contributions.
                    vec3 E = normalize( camera_center - vertex_worldspace );
                    vec3 result = vec3( 0.0 );
                    light_diffuse_contribution = vec3( 0.0 );
                    light_specular_contribution = vec3( 0.0 );
                    for(int i = 0; i < N_LIGHTS; i++){
                        // Lights store homogeneous coords - either a position or vector.  If w is 0, the 
                        // light will appear directional (uniform direction from all points), and we 
                        // simply obtain a vector towards the light by directly using the stored value.
                        // Otherwise if w is 1 it will appear as a point light -- compute the vector to 
                        // the point light's location from the current surface point.  In either case, 
                        // fade (attenuate) the light as the vector needed to reach it gets longer.  
                        vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                                       light_positions_or_vectors[i].w * vertex_worldspace;                                             
                        float distance_to_light = length( surface_to_light_vector );
        
                        vec3 L = normalize( surface_to_light_vector );
                        vec3 H = normalize( L + E );
                        // Compute the diffuse and specular components from the Phong
                        // Reflection Model, using Blinn's "halfway vector" method:
                        float diffuse  =      max( dot( N, L ), 0.0 );
                        float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                        float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                        
                        vec3 light_contribution = grass_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                        light_diffuse_contribution += attenuation * grass_color.xyz * light_colors[i].xyz * diffusivity * diffuse;
                        light_specular_contribution += attenuation * grass_color.xyz * specularity * specular;
                        result += attenuation * light_contribution;
                      }
                    return result;
                  }
        
            `;
    }

    vertex_glsl_code() {
        return this.shared_glsl_code() + `
                attribute vec3 position;
                attribute vec2 texture_coord;  
                attribute vec3 normal;                       
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                
                void main(){
                    worldPos = model_transform * vec4(position, 1.0);
                    float alpha = (layer / 10.0) * PerlinNoise3Pass(worldPos.xz + random(worldPos.xz) + vec2(time * 1.0, time * 1.0), 2.0) - 1.0;
                    gl_Position = projection_camera_model_transform * vec4(position.x + (0.02 * alpha), position.y + (0.04 * layer), position.z + (0.02 * alpha), 1.0);
                    f_tex_coord = texture_coord;
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                }`;
    }

    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                uniform sampler2D light_depth_texture;
                uniform float light_texture_size;
                uniform mat4 light_view_mat;
                uniform mat4 light_proj_mat;
                uniform float light_depth_bias;
                uniform bool draw_shadow;
                uniform bool fake_shadow_layer;
                
                float PCF_shadow(vec2 center, float projected_depth) {
                    float shadow = 0.0;
                    float texel_size = 1.0 / light_texture_size;
                    for(int x = -1; x <= 1; ++x)
                    {
                        for(int y = -1; y <= 1; ++y)
                        {
                            float light_depth_value = texture2D(light_depth_texture, center + vec2(x, y) * texel_size).r; 
                            shadow += (projected_depth >= light_depth_value + light_depth_bias) ? 0.8 : 0.0;        
                        }    
                    }
                    shadow /= 9.0;
                    return shadow;
                }
        
                void main(){
                    if ((worldPos.x < 14.0 && worldPos.x > -11.5) && (worldPos.z < 14.0 && worldPos.z > -11.5)){
                        discard;
                    }
                    if (fake_shadow_layer){
                        float perlin = 1.0 - (1.0 - PerlinNoise3Pass(vec2(worldPos.x + 0.03, worldPos.z - 0.03), 50.0)) * 2.2;
                        float white = 1.0 - (1.0 - perlinNoise(vec2(worldPos.x + 0.03, worldPos.z - 0.03))) * 40.0;
                        float alpha = perlin * white - ((layer + 0.2) * 1.8 / 1.0);
                        if (alpha < 0.0 || worldPos.y < -0.7){
                            discard;
                        }
                        gl_FragColor = vec4(grass_color.x * ambient * 1.3 + (layer / 200.0), grass_color.y * ambient * 1.3 + (layer / 200.0), grass_color.z * ambient * 1.3 + (layer / 200.0), 1.0);
                        vec4 tex_color = texture2D(texture, f_tex_coord);
                        if (tex_color.r > 0.0){
                            discard;
                        }
                    }
                    else {
                    if (layer > 0.0){
                        gl_FragColor = vec4(grass_color.x * ambient + (layer / 70.0), grass_color.y * ambient + (layer / 70.0), grass_color.z * ambient + (layer / 70.0), 1.0);
                    }
                    else {
                        gl_FragColor = vec4(ground_color.x * ambient, ground_color.y * ambient, ground_color.z * ambient, 1.0);
                    }
                    vec3 diffuse, specular;
                    vec3 other_than_ambient = phong_model_lights( normalize( N ), vertex_worldspace, diffuse, specular );
                    
                    if (draw_shadow) {
                        vec4 light_tex_coord = (light_proj_mat * light_view_mat * vec4(vertex_worldspace, 1.0));
                        light_tex_coord.xyz /= light_tex_coord.w; 
                        light_tex_coord.xyz *= 0.5;
                        light_tex_coord.xyz += 0.5;
                        float light_depth_value = texture2D( light_depth_texture, light_tex_coord.xy ).r;
                        float projected_depth = light_tex_coord.z;
                        
                        bool inRange =
                            light_tex_coord.x >= 0.0 &&
                            light_tex_coord.x <= 1.0 &&
                            light_tex_coord.y >= 0.0 &&
                            light_tex_coord.y <= 1.0;
                              
                        float shadowness = PCF_shadow(light_tex_coord.xy, projected_depth);
                        
                        if (inRange && shadowness > 0.3) {
                            diffuse *= 0.2 + 0.8 * (1.0 - shadowness);
                            specular *= 1.0 - shadowness;
                        }
                    }
                    
                    gl_FragColor.xyz += diffuse + specular;
                    
                    if (layer > 0.0){
                        float perlin = 1.0 - (1.0 - PerlinNoise3Pass(worldPos.xz, 50.0)) * 2.2;
                        float white = 1.0 - (1.0 - perlinNoise(worldPos.xz)) * 40.0;
                        float alpha = perlin * white - ((layer + 0.2) * 1.2 / 1.0);
                        if (alpha < 0.0 || worldPos.y < -0.7){
                            discard;
                        }
                        vec4 tex_color = texture2D(texture, f_tex_coord);
                        if (tex_color.r > 0.0){
                            discard;
                        }
                    }
                    }
                }`;
    }
}

//work in progress. I want to use a texture instead of procedural noise for the performance
class Grass_Shader_Background_Textured extends Shader {
    constructor(layer, num_lights = 2) {
        super();
        this.layer = layer;
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniform4fv(gpu_addresses.color, material.color);
        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000.0);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniform1f(gpu_addresses.layer, this.layer);

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

        if (material.PerlinTexture && material.PerlinTexture.ready && material.WhiteTexture && material.WhiteTexture.ready) {
            context.uniform1i(gpu_addresses.PerlinTexture, 0);
            material.PerlinTexture.activate(context);
            context.uniform1i(gpu_addresses.WhiteTexture, 1);
            material.WhiteTexture.activate(context, 1);
        }
    }

    shared_glsl_code() {
        return `precision mediump float;
        
                varying vec4 worldPos;
                uniform float time;
                uniform float layer;
                varying vec2 perlinTexPos, whiteTexPos;
                uniform sampler2D PerlinTexture;
                uniform sampler2D WhiteTexture;
                
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec3 squared_scale, camera_center;
                uniform vec4 color;
        
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
                        
                        vec3 light_contribution = color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                        result += attenuation * light_contribution;
                      }
                    return result;
                  }
                  
                float random (vec2 value){
                    return fract(sin(dot(value, vec2(94.8365, 47.053))) * 94762.9342);
                }
        
            `;
    }

    vertex_glsl_code() {
        return this.shared_glsl_code() + `
                attribute vec3 position;
                attribute vec3 normal;                       
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                attribute vec2 texture_coord;
                
                void main(){
                    worldPos = model_transform * vec4(position, 1.0);
                    whiteTexPos = texture_coord * 15.0;
                    perlinTexPos = texture_coord;
                    float alpha = 0.0;
                    gl_Position = projection_camera_model_transform * vec4(position, 1.0);
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                }`;
    }

    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                void main(){
                    gl_FragColor = vec4(color.x * ambient + (layer / 70.0), color.y * ambient + (layer / 70.0), color.z * ambient + (layer / 70.0), 1.0 - 0.0 * exp(-0.2 * (40.0 - distance(vec2(0,0), worldPos.xz))));
                    gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace);
                    
                    //if ((worldPos.x < 12.0 && worldPos.x > -12.0) && (worldPos.z < 12.0 && worldPos.z > -12.0)){
                    //    discard;
                   // }
                    //if (layer > 0.0){
                        vec4 perlinTex3 = texture2D(PerlinTexture, perlinTexPos);
                        vec4 whiteTex = texture2D(WhiteTexture, whiteTexPos);

                        //float perlin = 1.0 - (1.0 - perlinTex2.x + perlinTex1.x) * 2.2;
                        //float perlin = 1.0 - (1.0 - perlinTex2.x) * 2.0;
                        float perlin = perlinTex3.x;
                        float white = 1.0 - (1.0 - whiteTex.x) * 20.0;
                        float alpha = (perlin * white) - ((layer + 0.2) * 1.0 / 1.5);
                        gl_FragColor = perlinTex3;
                        //if (alpha < 0.0){
                        //    discard;
                        //}
                    //}
                }`;
    }
}

//work in progress. At the moment it just uses voronoi noise to make highlights and has very low opacity. Eventually gonna use
//depth pass, gerstner waves, tinting, fake specular, etc
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

    vertex_glsl_code() {
        return this.shared_glsl_code() + `
                attribute vec3 position;
                attribute vec3 normal;                         
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                
                void main(){
                    noisePos = (model_transform * vec4(position.x, position.y, position.z, 1.0)).xz * (1.0 / 1.3);
                    float noise = voronoi(noisePos);
                    gl_Position = projection_camera_model_transform * vec4( position.x, position.y + (noise / 4.0), position.z, 1.0 );
                    N = normalize( mat3( model_transform ) * vec3(normal.x, normal.y + (noise / 10.0), normal.z) / squared_scale);
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                }`;
    }

    //sets each pixel's color
    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                
                void main(){
                    gl_FragColor = vec4(shape_color.x * ambient, shape_color.y * ambient, shape_color.z * ambient, 0.1);
                    vec3 lighting = phong_model_lights( normalize( N ), vertex_worldspace);
                    float noise = min(pow(voronoi(noisePos), 4.0), pow(voronoi(noisePos), 6.0));
                    gl_FragColor.xyz += lighting + noise * lighting * 2.0 + noise * (ambient / 3.0);
                }`;
    }
}

//custom texture class that uses a software defined array as input instead of the html IMAGE object that the default class uses
//need to pass in the length and width of the data. If no data passed in, will create a black texture of the given size
//most of this is documented in the tiny-graphics.js file, so I will just comment the changes I made from that
class Dynamic_Texture extends Graphics_Card_Object {
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

class Buffered_Texture extends tiny.Graphics_Card_Object {
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

    drawOverrideMaterial(context, program_state, overrideMat){
        this.shape.draw(context, program_state, this.transform, overrideMat, this.renderArgs);
    }
}

class Base_Scene extends Scene {
    constructor() {
        super();

        //creates a blank custom texture for the grass occlusion
        this.grassOcclusionTexture = new Dynamic_Texture(256, 256);

        this.shapes = {
            'axis' : new defs.Axis_Arrows()
        };

        this.materials = {
            plastic: new Material(new defs.Phong_Shader(), {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            plain: new Material(new PlainShader()),
        };

        //member variables to track our interaction state
        this.isRaising = true;
        this.isLowering = false;
        this.isOccluding = false;

        this.lightDepthTextureSize = 2048;
        this.light_position = vec4(15, 8, -15, 0);
        this.light_color = color(5,5,5,0);
        this.light_view_target = vec4(0, 0, 0, 1);
        this.light_field_of_view = 130 * Math.PI / 180;
        this.light_view_mat = Mat4.look_at(
            vec3(this.light_position[0], this.light_position[1], this.light_position[2]),
            vec3(this.light_view_target[0], this.light_view_target[1], this.light_view_target[2]),
            vec3(0, 1, 0), // assume the light to target will have a up dir of +y, maybe need to change according to your case
        );
        this.light_proj_mat = Mat4.perspective(this.light_field_of_view, 1, 0.5, 500);

        //create the background grass plane. low density since we aren't deforming it
        this.background_grass_plane = new Scene_Object(new Triangle_Strip_Plane(20, 20, Vector3.create(0,0,0), 2),
            Mat4.scale(5,1,5), new Material(new Grass_Shader_Background(0), {grass_color: hex_color("#38af18"), ground_color: hex_color("#544101"),
                ambient: 0.2, diffusivity: 0.3, specularity: 0.032, smoothness: 100}), "TRIANGLE_STRIP");

        this.water_plane = new Scene_Object(new Triangle_Strip_Plane(5,5, Vector3.create(0,0,0), 5), Mat4.translation(-10,-0.7,-10).times(Mat4.scale(10,1,10)),
            new Material(new Phong_Water_Shader(), {color: hex_color("#4e6ef6"), ambient: 0.2, diffusivity: 0.7, specularity: 0.7, smoothness: 100}), "TRIANGLE_STRIP");

        //the main grass plane has a higher density since we want the deformation to look smooth
        this.grass_plane = new Scene_Object(new Triangle_Strip_Plane(26, 26, Vector3.create(0,0,0), 7),
            Mat4.translation(0,0,0), new Material(new Grass_Shader_Shadow(0), {grass_color: hex_color("#38af18"), ground_color: hex_color("#544101"),
                texture: this.grassOcclusionTexture, ambient: 0.2, diffusivity: 0.3, specularity: 0.032, smoothness: 100, fake_shadow_layer: false,
                light_depth_texture: null, lightDepthTextureSize: this.lightDepthTextureSize, draw_shadow: true, light_view_mat: this.light_view_mat, light_proj_mat: this.light_proj_mat}), "TRIANGLE_STRIP");

        //the skybox is just a sphere with the shader that makes the color look vaguely like sky above. We put everything inside this sphere
        this.skybox = new Scene_Object(new defs.Subdivision_Sphere(4), Mat4.scale(40, 40,40),
            new Material(new Skybox_Shader(), {top_color: hex_color("#268b9a"), mid_color: hex_color("#d1eaf6"), bottom_color: hex_color("#3d8f2b")}));

        this.init_ok = false;
    }

    display(context, program_state) {
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new Custom_Movement_Controls());
            program_state.set_camera(Mat4.look_at(vec3(7, 12, 23), vec3(1, 0, 0), vec3(0, 1, 0)));
            program_state.projection_transform = Mat4.perspective(Math.PI / 4, context.width / context.height, 1, 500);
        }
        program_state.lights = [new Light(this.light_position, this.light_color, 10000)];
    }
}

export class Test extends Base_Scene {
    constructor() {
        super();
    }

    make_control_panel(context) {
        this.key_triggered_button("raise terrain", ["r"], () => {
            this.isRaising = true;
            this.isLowering = false;
            this.isOccluding = false;
        });
        this.key_triggered_button("lower terrain", ["l"], () => {
            this.isRaising = false;
            this.isLowering = true;
            this.isOccluding = false;
        });
        this.key_triggered_button("occlude grass", ["o"], () => {
            this.isRaising = false;
            this.isLowering = false;
            this.isOccluding = true;
        });
    }

    texture_buffer_init(gl) {
        this.lightDepthTextureGPU = gl.createTexture();
        this.lightDepthTexture = new Buffered_Texture(this.lightDepthTextureGPU);

        gl.bindTexture(gl.TEXTURE_2D, this.lightDepthTextureGPU);
        gl.texImage2D(
            gl.TEXTURE_2D,      // target
            0,                  // mip level
            gl.DEPTH_COMPONENT, // internal format
            this.lightDepthTextureSize,   // width
            this.lightDepthTextureSize,   // height
            0,                  // border
            gl.DEPTH_COMPONENT, // format
            gl.UNSIGNED_INT,    // type
            null);              // data
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        // Depth Texture Buffer
        this.lightDepthFramebuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.lightDepthFramebuffer);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER,       // target
            gl.DEPTH_ATTACHMENT,  // attachment point
            gl.TEXTURE_2D,        // texture target
            this.lightDepthTextureGPU,         // texture
            0);                   // mip level
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        // create a color texture of the same size as the depth texture
        // see article why this is needed_
        this.unusedTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.unusedTexture);
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            this.lightDepthTextureSize,
            this.lightDepthTextureSize,
            0,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            null,
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        // attach it to the framebuffer
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER,        // target
            gl.COLOR_ATTACHMENT0,  // attachment point
            gl.TEXTURE_2D,         // texture target
            this.unusedTexture,         // texture
            0);                    // mip level
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
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
        let planeLoc = Vector.create(Math.ceil(planeLocPercent[0] * (plane.shape.length * plane.shape.density / 2)) + (plane.shape.length * plane.shape.density / 2),
            Math.ceil(planeLocPercent[1] * (plane.shape.width * plane.shape.density / 2)) + (plane.shape.width * plane.shape.density / 2));

        //attenuate strength based on brush radius so that large brushes don't raise terrain much faster than small brushes
        let strength = 0.05 / Math.max((brushRadius - 6), 1);
        for (let i = 5; i < brushRadius; i++) {
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

    resetFrameBuffer(context, program_state, projTransformStorage, cameraStorage) {
        context.context.bindFramebuffer(context.context.FRAMEBUFFER, null);
        context.context.viewport(0, 0, context.context.canvas.width, context.context.canvas.height);
        program_state.projection_transform = projTransformStorage;
        program_state.camera_inverse = cameraStorage;
    }

    render_scene(context, program_state, depthPass) {
        const t = program_state.animation_time;

        if(depthPass === false) {
            this.skybox.drawObject(context, program_state);
            this.shapes.axis.draw(context, program_state, Mat4.identity(), this.materials.plastic);

            //with the way the grass shader works, you need to draw it a number of times, each time specifying which level of the grass you want to draw
            //for the background 8-12 layers look good
            for (let i = 0; i < 12; i++) {
                this.background_grass_plane.material.shader.layer = i;
                this.background_grass_plane.drawObject(context, program_state);
            }

            //16 layers looks good for the main grass portion. can increase or decrease later if we want
            this.grass_plane.material.light_depth_texture = this.lightDepthTexture;
            this.grass_plane.material.draw_shadow = true;
            for (let i = 0; i < 16; i++) {
                this.grass_plane.material.shader.layer = i;
                this.grass_plane.drawObject(context, program_state);
                //this.grass_plane.material.fake_shadow_layer = true;
                //this.grass_plane.drawObject(context, program_state);
                //this.grass_plane.material.fake_shadow_layer = false;
            }

            this.water_plane.drawObject(context, program_state);
        }
        else{
            this.shapes.axis.draw(context, program_state, Mat4.identity(), this.materials.plain);
            this.background_grass_plane.drawOverrideMaterial(context, program_state, this.materials.plain);
            //this.grass_plane.drawOverrideMaterial(context, program_state, this.materials.plain);
            this.grass_plane.material.draw_shadow = false;
            for (let i = 0; i < 32; i++) {
                this.grass_plane.material.shader.layer = i;
                this.grass_plane.drawObject(context, program_state);
            }
        }
    }

    display(context, program_state) {
        super.display(context, program_state);

        if (!this.init_ok) {
            const ext = context.context.getExtension('WEBGL_depth_texture');
            if (!ext) {
                return alert('need WEBGL_depth_texture');  // eslint-disable-line
            }
            this.texture_buffer_init(context.context);

            this.init_ok = true;
        }

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
        this.render_scene(context, program_state, true);

        this.resetFrameBuffer(context, program_state, projTransformStorage, cameraStorage);
        this.render_scene(context, program_state, false);
    }
}