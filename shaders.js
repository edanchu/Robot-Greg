import {defs, tiny} from './examples/common.js';
const {vec3, vec4, vec, color, Matrix, Mat4, Light, Shape, Material, Shader, Texture, Scene} = tiny;

//im not gonna comment on how the shaders work for now. If you want to know how they work, ask me in person and ill go through them
//This shader creates the skybox by blending between 3 values, one for the horizon, under the horizon, and above the horizon
export class Skybox_Shader extends Shader {

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform));
        context.uniform4fv(gpu_addresses.top_color, material.top_color);
        context.uniform4fv(gpu_addresses.mid_color, material.mid_color);
        context.uniform4fv(gpu_addresses.bottom_color, material.bottom_color);
        context.uniform4fv(gpu_addresses.light_position, material.light_position);
    }
    shared_glsl_code() {
        return `precision mediump float;
               varying vec4 pos;
               varying vec4 worldPosition;
           `;
    }

    vertex_glsl_code() {
        return this.shared_glsl_code() + `
               attribute vec3 position;    
               uniform mat4 projection_camera_model_transform;
               uniform mat4 model_transform;
        
                void main(){
                    gl_Position = projection_camera_model_transform * vec4(position, 1.0);
                    pos = vec4(position, 1.0);
                    worldPosition = model_transform * vec4(position, 1.0);
                }`;
    }

    fragment_glsl_code() {
        return this.shared_glsl_code() + `
               uniform vec4 top_color;
               uniform vec4 mid_color;
               uniform vec4 bottom_color;
               uniform vec4 light_position;
                
                void main(){
                    float topGrad = pow(max(0.0, pos.y), 0.75);
                    float bottomGrad = pow(min(0.0, pos.y) * -1.0, 0.75);
                    vec4 midColor = (1.0 - (topGrad + bottomGrad)) * mid_color;
                    vec4 topColor = topGrad * top_color;
                    vec4 bottomColor = bottomGrad * bottom_color;
                    vec4 sunColor = vec4(0.0);
                    if (distance(worldPosition, light_position ) < 2.0) {
                        sunColor += vec4(1.0);
                    }
                    gl_FragColor = topColor + bottomColor + midColor + sunColor;
                }`;
    }
}

export class PlainShader extends Shader {

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

export class Grass_Shader_Shadow extends Shader {
    constructor(layer, num_lights = 2) {
        super();
        this.layer = layer;
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniformMatrix4fv(gpu_addresses.inv_transpose_model_transform, false, Matrix.flatten_2D_to_1D(Mat4.inverse(model_transform)));
        context.uniform4fv(gpu_addresses.grass_color, material.grass_color);
        context.uniform1i(gpu_addresses.ground_texture, 3);
        material.ground_texture.activate(context, 3);
        context.uniform1i(gpu_addresses.grass_coarse_texture, 4);
        material.grass_coarse_texture.activate(context, 4);
        context.uniform1i(gpu_addresses.grass_broad_texture, 5);
        material.grass_broad_texture.activate(context, 5);
        context.uniform1i(gpu_addresses.underwater_texture, 6);
        material.underwater_texture.activate(context, 6);
        context.uniform1i(gpu_addresses.lush_grass, material.lush_grass);

        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000.0);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniform1f(gpu_addresses.layer, this.layer);
        context.uniform1i(gpu_addresses.texture, 0);
        material.texture.activate(context, 0);
        context.uniform1f(gpu_addresses.ambient, material.ambient);
        context.uniform1f(gpu_addresses.diffusivity, material.diffusivity);
        context.uniform1f(gpu_addresses.specularity, material.specularity);
        context.uniform1f(gpu_addresses.smoothness, material.smoothness);
        const O = vec4(0, 0, 0, 1), camera_center = graphics_state.camera_transform.times(O).to3();
        context.uniform3fv(gpu_addresses.camera_center, camera_center);

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
    
        context.uniformMatrix4fv(gpu_addresses.light_mat, false, Matrix.flatten_2D_to_1D(material.light_proj_mat.times(material.light_view_mat).transposed()));
        context.uniform1i(gpu_addresses.draw_shadow, material.draw_shadow);
        context.uniform1f(gpu_addresses.light_depth_bias, 0.35);
        context.uniform1f(gpu_addresses.light_texture_size, material.lightDepthTextureSize);
        context.uniform1i(gpu_addresses.light_depth_texture, 1);
        if (material.draw_shadow === true) {
            material.light_depth_texture.activate(context, 1);
        }
    }

    shared_glsl_code() {
        return `precision mediump float;
        
                varying vec4 worldPos;
                uniform float time;
                uniform float layer;
                
                uniform sampler2D texture;
                uniform sampler2D ground_texture;
                uniform sampler2D underwater_texture;
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
                uniform vec3 camera_center;
                uniform vec4 grass_color;
        
                varying vec3 N, vertex_worldspace;
                varying vec3 diffuse, specular;
                
                vec3 triPlanar(vec3 N, vec3 vertex_worldspace, sampler2D texture){
                    vec3 x = texture2D(texture, vertex_worldspace.zy / 2.0).xyz;
                    vec3 y = texture2D(texture, vertex_worldspace.xz / 2.0).xyz;
                    vec3 z = texture2D(texture, vertex_worldspace.xy / 2.0).xyz;
                    vec3 normal = normalize(abs(N));
                    vec3 normalWeight = normal / (normal.x + normal.y + normal.z);
                    return normalWeight.x * x + normalWeight.y * y + normalWeight.z * z;
                }
                
                vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace, 
                        out vec3 light_diffuse_contribution, out vec3 light_specular_contribution){                                        
                    vec3 E = normalize( camera_center - vertex_worldspace );
                    vec3 result = vec3( 0.0 );
                    light_diffuse_contribution = vec3( 0.0 );
                    light_specular_contribution = vec3( 0.0 );
                    for(int i = 0; i < N_LIGHTS; i++){
                        vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                                       light_positions_or_vectors[i].w * vertex_worldspace;                                             
                        float distance_to_light = length( surface_to_light_vector );
        
                        vec3 L = normalize( surface_to_light_vector );
                        vec3 H = normalize( L + E );
                        float diffuse  =      min(max( dot( N, L ), 0.2 ), 0.5);
                        float specular = pow( max( dot( N, H ), 0.2 ), smoothness );
                        float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                        
                        vec3 light_contribution = vec3(0.0,0.0,0.0);
                        if (layer > 0.0){
                            light_contribution = grass_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                            light_diffuse_contribution += attenuation * grass_color.xyz * light_colors[i].xyz * diffusivity * diffuse;
                            light_specular_contribution += attenuation * grass_color.xyz * specularity * specular;
                        }
                        else{
                            vec4 dirtTexColor = vec4(triPlanar(N, vertex_worldspace, ground_texture), 1.0);
                            vec4 underwaterTexColor = vec4(triPlanar(N, vertex_worldspace * 1.3, underwater_texture), 1.0);
                            vec4 groundTexColor = mix(dirtTexColor, underwaterTexColor, (vertex_worldspace.y < 0.0) ? 1.0 - min(1.1, 1.1 - abs(vertex_worldspace.y)): 0.0);
                            light_contribution = groundTexColor.xyz * light_colors[i].xyz * diffusivity * diffuse + light_colors[i].xyz * specularity * specular;
                            light_diffuse_contribution += attenuation * groundTexColor.xyz / 3.0 * light_colors[i].xyz * diffusivity * diffuse;
                            light_specular_contribution += attenuation * groundTexColor.xyz * specularity * specular;
                        }
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
                uniform mat4 inv_transpose_model_transform;
                
                void main(){
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                    float alpha = (layer / 10.0) * PerlinNoise3Pass(vertex_worldspace.xz + vec2(time * 1.5, time * 1.5), 2.0);
                    gl_Position = projection_camera_model_transform * vec4(position.x + (0.2 * alpha), position.y + (0.04 * layer), position.z + (0.2 * alpha), 1.0);
                    f_tex_coord = texture_coord;
                    N =  normalize((inv_transpose_model_transform * vec4(normal, 0.0)).xyz);
                    vec3 other_than_ambient = phong_model_lights( N, vertex_worldspace, diffuse, specular);
                }`;
    }

    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                uniform sampler2D light_depth_texture;
                uniform float light_texture_size;
                uniform mat4 light_mat;
                uniform float light_depth_bias;
                uniform bool draw_shadow;
                uniform sampler2D grass_coarse_texture;
                uniform sampler2D grass_broad_texture;
                uniform bool lush_grass;
                
                float linearDepth(float val){
                    val = 2.0 * val - 1.0;
                    return (2.0 * 0.5 * 150.0) / (150.0 + 0.5 - val * (150.0 - 0.5));
                }
                
                
                float PCF_shadow(vec2 center, float projected_depth) {
                    float shadow = 0.0;
                    float texel_size = 1.0 / light_texture_size;
                    for(int x = -1; x <= 1; ++x)
                    {
                        for(int y = -1; y <= 1; ++y)
                        {
                            float light_depth_value = linearDepth(texture2D(light_depth_texture, center + vec2(x, y) * texel_size).r); 
                            shadow += (linearDepth(projected_depth) >= light_depth_value + light_depth_bias) ? 0.8 : 0.0;        
                        }    
                    }
                    shadow /= 9.0;
                    return shadow;
                }
        
                void main(){
                    if (layer > 0.0){
                        gl_FragColor = vec4(grass_color.xyz * ambient + (layer / 70.0), 1.0);
                    }
                    else {
                        vec4 dirtTexColor = vec4(triPlanar(N, vertex_worldspace, ground_texture), 1.0);
                        vec4 underwaterTexColor = vec4(triPlanar(N, vertex_worldspace * 1.3, underwater_texture), 1.0);
                        vec4 groundTexColor = mix(dirtTexColor, underwaterTexColor, (vertex_worldspace.y < 0.0) ? 1.0 - min(1.1, 1.1 - abs(vertex_worldspace.y)): 0.0);
                        gl_FragColor = vec4(groundTexColor.xyz * ambient * 3.0, 1.0);
                    }
                    vec3 diffuse1 = diffuse;
                    vec3 specular1 = specular;
                    if (draw_shadow) {
                        vec4 light_tex_coord = (light_mat * vec4(vertex_worldspace, 1.0));
                        light_tex_coord.xyz /= light_tex_coord.w; 
                        light_tex_coord.xyz *= 0.5;
                        light_tex_coord.xyz += 0.5;
                        float projected_depth = light_tex_coord.z;
                        
                        bool inRange =
                            light_tex_coord.x >= 0.0 &&
                            light_tex_coord.x <= 1.0 &&
                            light_tex_coord.y >= 0.0 &&
                            light_tex_coord.y <= 1.0;
                                                          
                        float shadowness = PCF_shadow(light_tex_coord.xy, projected_depth);
                        
                        if (inRange && shadowness > 0.3) {
                            diffuse1 *= 0.1 + 0.9 * (1.0 - shadowness);
                            specular1 *= 1.0 - shadowness * 1.2;
                        }
                    }
                    
                    gl_FragColor.xyz += diffuse1 + specular1;
                    
                    if (layer > 0.0){
                        if(vertex_worldspace.y < -1.0) discard;
                        // float perlin = 1.0 - (1.0 - PerlinNoise3Pass(vertex_worldspace.xz, 50.0)) * 2.2;
                        // float white = 1.0 - (1.0 - perlinNoise(vertex_worldspace.xz)) * 40.0;
                        // float alpha = perlin * white - ((layer + 0.2) * 1.2 / 1.0);
                        // if (alpha < 0.0 || vertex_worldspace.y < -1.0){
                        //     discard;
                        // }
                        
                        float coarse = texture2D(grass_coarse_texture, f_tex_coord / 1.2).x * 2.0 - 1.1;
                        float broad = 0.0;
                        coarse = coarse * 1.2;
                        if (!lush_grass){
                            broad = 1.0 - (1.0 - perlinNoise(vertex_worldspace.xz)) * 40.0;
                        }
                        else{
                            broad = texture2D(grass_broad_texture, f_tex_coord * 2.0).x * 2.0 - 1.1;
                            broad = 1.0 - (1.0 - broad) * 25.0;
                        }
                        
                        float alpha =  broad * coarse - ((layer + 0.2) * 1.1 / 1.0);
                        
                        if (alpha < 0.0){
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

export class Grass_Shader_Background_Shadow extends Shader {
    constructor(layer, num_lights = 2) {
        super();
        this.layer = layer;
        this.num_lights = num_lights;
        this.sentOnce = false;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniformMatrix4fv(gpu_addresses.inv_transpose_model_transform, false, Matrix.flatten_2D_to_1D(Mat4.inverse(model_transform)));
        context.uniform4fv(gpu_addresses.grass_color, material.grass_color);
        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000.0);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniform1f(gpu_addresses.layer, this.layer);
        context.uniform1i(gpu_addresses.ground_texture, 3);
        material.ground_texture.activate(context, 3);
        context.uniform1i(gpu_addresses.grass_coarse_texture, 4);
        material.grass_coarse_texture.activate(context, 4);
        context.uniform1i(gpu_addresses.grass_broad_texture, 5);
        material.grass_broad_texture.activate(context, 5);

        context.uniform1f(gpu_addresses.ambient, material.ambient);
        context.uniform1f(gpu_addresses.diffusivity, material.diffusivity);
        context.uniform1f(gpu_addresses.specularity, material.specularity);
        context.uniform1f(gpu_addresses.smoothness, material.smoothness);
        const O = vec4(0, 0, 0, 1), camera_center = graphics_state.camera_transform.times(O).to3();
        context.uniform3fv(gpu_addresses.camera_center, camera_center);
        context.uniform1i(gpu_addresses.lush_grass, material.lush_grass);
    
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

        context.uniformMatrix4fv(gpu_addresses.light_mat, false, Matrix.flatten_2D_to_1D(material.light_proj_mat.times(material.light_view_mat).transposed()));
        context.uniform1i(gpu_addresses.draw_shadow, material.draw_shadow);
        context.uniform1f(gpu_addresses.light_depth_bias, 0.35);
        context.uniform1f(gpu_addresses.light_texture_size, material.lightDepthTextureSize);
        context.uniform1i(gpu_addresses.light_depth_texture, 1);
        if (material.draw_shadow === true) {
            material.light_depth_texture.activate(context, 1);
        }
    }

    shared_glsl_code() {
        return `precision mediump float;
        
                varying vec4 worldPos;
                uniform float time;
                uniform float layer;
                
                uniform sampler2D ground_texture;
                varying vec2 f_tex_coord;
                varying vec3 diffuse, specular;
                                
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
                uniform vec3 camera_center;
                uniform vec4 grass_color;
        
                varying vec3 N, vertex_worldspace;
                vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace, 
                        out vec3 light_diffuse_contribution, out vec3 light_specular_contribution){                                        
                    vec3 E = normalize( camera_center - vertex_worldspace );
                    vec3 result = vec3( 0.0 );
                    light_diffuse_contribution = vec3( 0.0 );
                    light_specular_contribution = vec3( 0.0 );
                    for(int i = 0; i < N_LIGHTS; i++){
                        vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                                       light_positions_or_vectors[i].w * vertex_worldspace;                                             
                        float distance_to_light = length( surface_to_light_vector );
        
                        vec3 L = normalize( surface_to_light_vector );
                        vec3 H = normalize( L + E );
                        float diffuse  =      max( dot( N, L ), 0.0 );
                        float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                        float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                        
                        vec3 light_contribution = vec3(1.0,1.0,1.0);
                        if (layer > 0.0){
                            light_contribution = grass_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                            light_diffuse_contribution += attenuation * grass_color.xyz * light_colors[i].xyz * diffusivity * diffuse;
                            light_specular_contribution += attenuation * grass_color.xyz * specularity * specular;
                        }
                        else{
                            vec4 groundTexColor = texture2D(ground_texture, f_tex_coord * 60.0);
                            light_contribution = groundTexColor.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                            light_diffuse_contribution += attenuation * groundTexColor.xyz / 3.0 * light_colors[i].xyz * diffusivity * diffuse;
                            light_specular_contribution += attenuation * groundTexColor.xyz / 3.0 * specularity * specular;
                        }
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
                attribute vec2 texture_coord;
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                uniform mat4 inv_transpose_model_transform;
                
                void main(){
                    worldPos = model_transform * vec4(position, 1.0);
                    float alpha = (layer / 10.0) * PerlinNoise3Pass(worldPos.xz + vec2(time * 1.5, time * 1.5), 2.0);
                    gl_Position = projection_camera_model_transform * vec4(position.x + (0.02 * alpha), position.y + (0.04 * layer), position.z + (0.02 * alpha), 1.0);
                    N =  normalize(inv_transpose_model_transform * vec4(normal, 0.0)).xyz;
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                    f_tex_coord = texture_coord;
                    vec3 other_than_ambient = phong_model_lights( N, vertex_worldspace, diffuse, specular);
                }`;
    }

    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                uniform sampler2D light_depth_texture;
                uniform float light_texture_size;
                uniform mat4 light_mat;
                uniform float light_depth_bias;
                uniform bool draw_shadow;
                uniform sampler2D grass_coarse_texture;
                uniform sampler2D grass_broad_texture;
                uniform bool lush_grass;
                
                float linearDepth(float val){
                    val = 2.0 * val - 1.0;
                    return (2.0 * 0.5 * 150.0) / (150.0 + 0.5 - val * (150.0 - 0.5));
                }
                
                float PCF_shadow(vec2 center, float projected_depth) {
                    float shadow = 0.0;
                    float texel_size = 1.0 / light_texture_size;
                    for(int x = -1; x <= 1; ++x)
                    {
                        for(int y = -1; y <= 1; ++y)
                        {
                            float light_depth_value = linearDepth(texture2D(light_depth_texture, center + vec2(x, y) * texel_size).r); 
                            shadow += (linearDepth(projected_depth) >= light_depth_value + light_depth_bias) ? 0.8 : 0.0;        
                        }    
                    }
                    shadow /= 9.0;
                    return shadow;
                }
                
                void main(){
                    if ((worldPos.x < 13.5 && worldPos.x > -11.5) && (worldPos.z < 13.5 && worldPos.z > -11.5)){
                        discard;
                    }
                    if (layer > 0.0){
                        gl_FragColor = vec4(grass_color.x * ambient + (layer / 70.0), grass_color.y * ambient + (layer / 70.0), grass_color.z * ambient + (layer / 70.0), 1.0 - exp(-0.5 * (40.0 - distance(vec4(0,0,0,0), worldPos))));
                    }
                    else {
                        vec4 groundTexColor = texture2D(ground_texture, f_tex_coord * 60.0);
                        gl_FragColor = vec4(groundTexColor.xyz * ambient * 3.0, 1.0 - exp(-0.5 * (40.0 - distance(vec4(0,0,0,0), worldPos))));
                    }
                    vec3 diffuse1 = diffuse;
                    vec3 specular1 = specular;
                    
                    if (draw_shadow) {
                        vec4 light_tex_coord = (light_mat * vec4(vertex_worldspace, 1.0));
                        light_tex_coord.xyz /= light_tex_coord.w; 
                        light_tex_coord.xyz *= 0.5;
                        light_tex_coord.xyz += 0.5;
                        float projected_depth = light_tex_coord.z;
                        
                        bool inRange =
                            light_tex_coord.x >= 0.0 &&
                            light_tex_coord.x <= 1.0 &&
                            light_tex_coord.y >= 0.0 &&
                            light_tex_coord.y <= 1.0;
                                                          
                        float shadowness = PCF_shadow(light_tex_coord.xy, projected_depth);
                        
                        if (inRange && shadowness > 0.3) {
                            diffuse1 *= 0.1 + 0.9 * (1.0 - shadowness);
                            specular1 *= 1.0 - shadowness * 1.2;
                        }
                    }
                    
                    gl_FragColor.xyz += diffuse1 + specular1;

                    if (layer > 0.0){
                        if (worldPos.y < -1.0){
                             discard;
                        }
                        
                        float coarse = texture2D(grass_coarse_texture, 5.0 * f_tex_coord / 1.2).x * 2.0 - 1.1;
                        float broad = 0.0;
                        coarse = coarse * 1.2;
                        if (!lush_grass){
                            broad = 1.0 - (1.0 - perlinNoise(worldPos.xz)) * 40.0;
                        }
                        else{
                            broad = texture2D(grass_broad_texture, f_tex_coord * 3.0).x * 2.0 - 1.0;
                            broad = 1.0 - (1.0 - broad) * 25.0;
                        }
                        
                        float alpha =  broad * coarse - ((layer + 0.2) * 1.1);
                        
                        if (alpha < 0.0 || worldPos.y < -1.0){
                             discard;
                        }
                    }
                }`;
    }
}

export class Water_Shader extends Shader{

    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniformMatrix4fv(gpu_addresses.inv_transpose_model_transform, false, Matrix.flatten_2D_to_1D(Mat4.inverse(model_transform)));
        context.uniformMatrix4fv(gpu_addresses.inv_transpose_view_matrix, false, Matrix.flatten_2D_to_1D(Mat4.inverse(graphics_state.camera_inverse)));
        context.uniformMatrix4fv(gpu_addresses.view_matrix, false, Matrix.flatten_2D_to_1D(graphics_state.camera_inverse.transposed()));
        context.uniformMatrix4fv(gpu_addresses.proj_matrix, false, Matrix.flatten_2D_to_1D(graphics_state.projection_transform.transposed()));
        context.uniform1f(gpu_addresses.doSSR, material.doSSR);
        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000);
        context.uniform4fv(gpu_addresses.shallow_color, material.shallow_color);
        context.uniform4fv(gpu_addresses.deep_color, material.deep_color);
        context.uniform1f(gpu_addresses.ambient, material.ambient);
        context.uniform1f(gpu_addresses.diffusivity, material.diffusivity);
        context.uniform1f(gpu_addresses.specularity, material.specularity);
        context.uniform1f(gpu_addresses.smoothness, material.smoothness);
        const O = vec4(0, 0, 0, 1), camera_center = graphics_state.camera_transform.times(O).to3();
        context.uniform3fv(gpu_addresses.camera_center, camera_center);

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
    
        material.depth_texture.activate(context, 1);
        context.uniform1i(gpu_addresses.depth_texture, 1);

        context.uniform1i(gpu_addresses.bg_color_texture, 2);
        material.bg_color_texture.activate(context, 2);
    
        context.uniform1i(gpu_addresses.derivative_height, 4);
        material.derivative_height.activate(context, 4);
    
        context.uniform1i(gpu_addresses.water_flow, 5);
        material.water_flow.activate(context, 5);
    }

    shared_glsl_code() {
        return `precision mediump float;
                uniform float time;
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec3 camera_center;
                uniform vec4 shallow_color;
                uniform vec4 deep_color;
                varying vec2 f_texture_coord;
                varying vec3 N, vertex_worldspace;
                
                
                float random (vec2 value){
                    return fract(sin(dot(value, vec2(94.8365, 47.053))) * 94762.9342);
                }

                float lerp(float a, float b, float percent){
                    return (1.0 - percent) * a + (percent * b);
                }
                
                vec2 voronoi(vec2 pos){
                  vec2 baseCell = floor(pos);
                  float minDist = 10.0;
                  vec2 center = vec2(0.0, 0.0);
                  
                  for (int i = -1; i <= 1; i++){
                    for (int j = -1; j <= 1; j++){
                      vec2 currentCell = baseCell + vec2(i, j);
                      vec2 posInCell = currentCell + random(currentCell);
                      center = distance(posInCell, pos) < minDist ? currentCell : center;
                      minDist = min(distance(posInCell, pos), minDist);
                    }
                  }
                  return center;
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
                        
                        vec3 light_contribution = vec3(1.0) * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                        result += attenuation * light_contribution;
                      }
                    return result;
                  }
                  
                //   vec3 GerstnerWave (vec4 wave, vec3 p, inout vec3 tangent, inout vec3 binormal, float timeOffset) {
                //     float steepness = wave.z;
                //     float wavelength = wave.w;
                //     float k = 2.0 * 3.141592653589 / wavelength;
                //     float c = sqrt(9.8 / k);
                //     vec2 d = normalize(wave.xy);
                //     float f = k * (dot(d, p.xz) - c * time / timeOffset);
                //     float a = steepness / k;
                //
                //     tangent += vec3(
                //        -d.x * d.x * (steepness * sin(f)),
                //         d.x * (steepness * cos(f)),
                //        -d.x * d.y * (steepness * sin(f)));
                //     binormal += vec3(
                //         -d.x * d.y * (steepness * sin(f)),
                //         d.y * (steepness * cos(f)),
                //         -d.y * d.y * (steepness * sin(f)));
                //     return vec3(
                //         d.x * (a * cos(f)),
                //         a * sin(f),
                //         d.y * (a * cos(f)));
                // }
            `;
    }

    vertex_glsl_code() {
        return this.shared_glsl_code() + `
                attribute vec3 position;
                attribute vec3 normal;
                attribute vec2 texture_coord;
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                uniform mat4 view_matrix;
                varying vec3 vertex_viewspace;
                uniform mat4 inv_transpose_view_matrix;
                uniform mat4 inv_transpose_model_transform;
                
                void main(){
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                    vertex_viewspace = (view_matrix * model_transform * vec4( position, 1.0 )).xyz;
                    gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                    N =  normalize(inv_transpose_view_matrix * inv_transpose_model_transform * vec4(normal, 0.0)).xyz;
                    f_texture_coord = texture_coord;
                }`;
    }

    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                uniform sampler2D depth_texture;
                uniform sampler2D bg_color_texture;
                uniform sampler2D water_flow;
                uniform sampler2D derivative_height;
                uniform mat4 proj_matrix;
                uniform mat4 model_transform;
                uniform mat4 view_matrix;
                varying vec3 vertex_viewspace;
                uniform mat4 inv_transpose_view_matrix;
                uniform mat4 inv_transpose_model_transform;
                uniform bool doSSR;
                
                 float linearDepth(float val){
                    val = 2.0 * val - 1.0;
                    return (2.0 * 1.0 * 150.0) / (150.0 + 1.0 - val * (150.0 - 1.0));
                }
                
                vec3 binSearch(inout vec3 pos, inout vec3 dir, inout float delta){
                    vec4 coords;
                    float depth;
                    for (float i = 0.0; i < 10.0; i++){
                         dir *= 1.1;
                         coords = proj_matrix * vec4(pos,1.0);
                         coords.xy = (coords.xy/coords.w) * 0.5 + 0.5;
                         depth = linearDepth(texture2D(depth_texture, coords.xy).r);

                        delta = pos.z - depth;
                        if (delta < 0.0){
                            pos += dir;
                        }
                        else{
                            pos -= dir;
                        }

                    }
                    coords = proj_matrix * vec4(pos,1.0);
                    coords.xy = (coords.xy/coords.w) * 0.5 + 0.5;
                    return vec3(coords.xy, depth);
                }
                
                vec3 Distort (vec2 uv, vec2 flowVector, vec2 jump, float flowOffset, float tiling, float time, bool flowB) {
                    float phaseOffset = flowB ? 0.5 : 0.0;
                    float progress = fract(time + phaseOffset);
                    vec3 uvw;
                    uvw.xy = uv - flowVector * (progress + flowOffset);
                    uvw.xy *= tiling;
                    uvw.xy += phaseOffset;
                    uvw.xy += (time - progress) * jump;
                    uvw.z = 1.0 - abs(1.0 - 2.0 * progress);
                    return uvw;
                }
                
                vec3 UnpackDerivativeHeight(vec4 textureData) {
                    vec3 dh = textureData.agb;
                    dh.xy = dh.xy * 2.0 - 1.0;
                    return dh;
                }

                void main(){
                    vec3 flow = texture2D(water_flow, f_texture_coord).xyz;
                    flow.xy = flow.xy * 2.0 - 1.0;
                    flow *= 0.3;
                    vec3 uvwA = Distort(f_texture_coord, flow.xy, vec2(0.24), -0.5, 9.0, time / 45.0, false);
                    vec3 uvwB = Distort(f_texture_coord, flow.xy, vec2(0.208333), -0.5, 9.0, time / 45.0, true);
                    float heightScale = flow.z * 0.25 + 0.75;
                    vec3 dhA = UnpackDerivativeHeight(texture2D(derivative_height, uvwA.xy)) * uvwA.z * heightScale;
                    vec3 dhB = UnpackDerivativeHeight(texture2D(derivative_height, uvwB.xy)) * uvwB.z * heightScale;
                    vec3 normal = normalize(vec3(-(dhA.xy + dhB.xy), 1.0));
                    
                    float refractionStrength = 0.036;
                    vec2 bgSS = vec2((gl_FragCoord.x - 0.5) / 1279.0, (gl_FragCoord.y - 0.5) / 719.0);
                    float refractionDepthVal = texture2D(depth_texture, bgSS + (normal.xy * refractionStrength)).r;
                    refractionDepthVal = linearDepth(refractionDepthVal) - linearDepth(gl_FragCoord.z);
                    if (refractionDepthVal > 0.0)
                        bgSS += normal.xy * refractionStrength;
                    
                    
                    
                    vec4 bgColor = texture2D(bg_color_texture, bgSS);
                    float depthVal = texture2D(depth_texture, bgSS).r;
                    float depthDifference = linearDepth(depthVal) - linearDepth(gl_FragCoord.z);
                    
                    vec3 reflectColor = vec3(0.0);
                    if (doSSR){
                    vec3 reflectDirection = reflect(normalize(vertex_viewspace), normalize((inv_transpose_view_matrix * inv_transpose_model_transform * vec4(0.0, 1.0, 0.0, 0.0)).xyz));
                    vec3 search = vec3(0.0);
                    bool reflectFound = false;
                    for(float i = 0.0; i < 5.0; i++){
                        if (reflectDirection.z > 0.0) break;
                        reflectDirection *= 2.0;
                        vec3 testPoint = vertex_viewspace + reflectDirection;
                        vec4 testPosSS = (proj_matrix * vec4(testPoint, 1.0));
                        testPosSS.xy = ( testPosSS.xy / testPosSS.w) * 0.5 + 0.5;
                        if (testPosSS.x >= 1.0 || testPosSS.y >= 1.0) break;
                        float testPointDepth = linearDepth(texture2D(depth_texture, testPosSS.xy).r);
                        float diff = testPoint.z - testPointDepth;
                        if (diff < 1.2){
                            reflectFound = true;
                            reflectDirection = normalize(reflectDirection);
                            search = binSearch(testPoint, reflectDirection, diff);
                            break;
                        }
                    }
                    if (reflectFound == true){
                        vec2 edgeFadeVec = smoothstep(0.2, 0.6, abs(vec2(0.5, 0.5) - search.xy));
                        float edgeFade = clamp(1.0 - (edgeFadeVec.x + edgeFadeVec.y), 0.0, 1.0);
                        reflectColor += texture2D(bg_color_texture, search.xy + (normal.xy * refractionStrength)).xyz * edgeFade;
                    }
                    }
                    
                    vec3 lighting = phong_model_lights( normalize( normal ), vertex_worldspace);
                    
                    float foam = 0.0;
                    if (depthDifference > 0.1){
                        foam = ((2.5 + sin(-depthDifference * 10.0 + time * 2.0)) / 2.0) * (pow(2.0, -10.0 * depthDifference));
                    }
                   
                    gl_FragColor = mix(mix(shallow_color, deep_color, (sin(min(depthDifference / 5.0, 1.0) * 3.14159 / 2.0 ))), bgColor, 1.0 - (sin(max(depthDifference / 8.0, 0.0) * 3.14159 / 2.0 )));
                    gl_FragColor.xyz += lighting + foam;
                    gl_FragColor.xyz += reflectColor;
                    // gl_FragColor = vec4(reflectColor, 1.0);
                }`;
    }
}

export class Shadow_Textured_Phong extends Shader {
    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000.0);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniformMatrix4fv(gpu_addresses.inv_transpose_model_transform, false, Matrix.flatten_2D_to_1D(Mat4.inverse(model_transform)));
        context.uniform1i(gpu_addresses.color_texture, 0);
        material.color_texture.activate(context, 0);

        context.uniform1f(gpu_addresses.ambient, material.ambient);
        context.uniform1f(gpu_addresses.diffusivity, material.diffusivity);
        context.uniform1f(gpu_addresses.specularity, material.specularity);
        context.uniform1f(gpu_addresses.smoothness, material.smoothness);
        const O = vec4(0, 0, 0, 1), camera_center = graphics_state.camera_transform.times(O).to3();
        context.uniform3fv(gpu_addresses.camera_center, camera_center);

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
        context.uniform1f(gpu_addresses.light_depth_bias, 0.3);
        context.uniform1f(gpu_addresses.light_texture_size, material.lightDepthTextureSize);
        context.uniform1i(gpu_addresses.light_depth_texture, 1);
        if (material.draw_shadow === true) {
            material.light_depth_texture.activate(context, 1);
        }
    }

    shared_glsl_code() {
        return `precision mediump float;
        
                uniform float time;
                
                uniform sampler2D color_texture;
                varying vec2 f_tex_coord;
                
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec3 camera_center;
        
                varying vec3 N, vertex_worldspace;
                vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace, 
                        out vec3 light_diffuse_contribution, out vec3 light_specular_contribution ){                                        
                    vec3 E = normalize( camera_center - vertex_worldspace );
                    vec3 result = vec3( 0.0 );
                    light_diffuse_contribution = vec3( 0.0 );
                    light_specular_contribution = vec3( 0.0 );
                    for(int i = 0; i < N_LIGHTS; i++){
                        vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                                       light_positions_or_vectors[i].w * vertex_worldspace;                                             
                        float distance_to_light = length( surface_to_light_vector );
        
                        vec3 L = normalize( surface_to_light_vector );
                        vec3 H = normalize( L + E );
                        float diffuse  =      max( dot( N, L ), 0.0 );
                        float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                        float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                        
                        vec3 light_contribution = vec3(0.5,0.5,0.5) * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                        light_diffuse_contribution += attenuation * vec3(0.5,0.5,0.5) * light_colors[i].xyz * diffusivity * diffuse;
                        light_specular_contribution += attenuation * vec3(0.5,0.5,0.5) * specularity * specular;
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
                uniform mat4 inv_transpose_model_transform;
                
                void main(){
                    gl_Position = projection_camera_model_transform * vec4(position, 1.0);
                    f_tex_coord = texture_coord;
                    N =  normalize(inv_transpose_model_transform * vec4(normal, 0.0)).xyz;
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
                
                float linearDepth(float val){
                    val = 2.0 * val - 1.0;
                    return (2.0 * 0.5 * 150.0) / (150.0 + 0.5 - val * (150.0 - 0.5));
                }
                
                float PCF_shadow(vec2 center, float projected_depth) {
                    float shadow = 0.0;
                    float texel_size = 1.0 / light_texture_size;
                    for(int x = -1; x <= 1; ++x)
                    {
                        for(int y = -1; y <= 1; ++y)
                        {
                            float light_depth_value = linearDepth(texture2D(light_depth_texture, center + vec2(x, y) * texel_size).r); 
                            shadow += (linearDepth(projected_depth) >= light_depth_value + light_depth_bias) ? 0.8 : 0.0;        
                        }    
                    }
                    shadow /= 9.0;
                    return shadow;
                }
        
                void main(){
                    vec4 color = texture2D(color_texture, f_tex_coord);
                    
                    gl_FragColor = vec4(color.xyz * ambient, color.w);
                    
                    vec3 diffuse, specular;
                    vec3 other_than_ambient = phong_model_lights( normalize( N ), vertex_worldspace, diffuse, specular);
                    
                    if (draw_shadow) {
                        vec4 light_tex_coord = (light_proj_mat * light_view_mat * vec4(vertex_worldspace, 1.0));
                        light_tex_coord.xyz /= light_tex_coord.w; 
                        light_tex_coord.xyz *= 0.5;
                        light_tex_coord.xyz += 0.5;
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
                    gl_FragColor.xyz += abs(diffuse) + abs(specular);

                }`;
    }
}

export class Shadow_Textured_Phong_Maps extends Shader{
    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000.0);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniformMatrix4fv(gpu_addresses.inv_transpose_model_transform, false, Matrix.flatten_2D_to_1D(Mat4.inverse(model_transform)));
        context.uniform1i(gpu_addresses.color_texture, 0);
        material.color_texture.activate(context, 0);

        context.uniform1i(gpu_addresses.specular_texture, 2);
        material.color_texture.activate(context, 2);
        
        context.uniform1i(gpu_addresses.normal_texture, 3);
        if (material.normal_texture != null) {
            material.normal_texture.activate(context, 3);
            context.uniform1i(gpu_addresses.use_normal, true);
        }
        else{
            context.uniform1i(gpu_addresses.use_normal, false);
        }

        context.uniform1f(gpu_addresses.ambient, material.ambient);
        context.uniform1f(gpu_addresses.diffusivity, material.diffusivity);
        context.uniform1f(gpu_addresses.specularity, material.specularity);
        context.uniform1f(gpu_addresses.smoothness, material.smoothness);
        const O = vec4(0, 0, 0, 1), camera_center = graphics_state.camera_transform.times(O).to3();
        context.uniform3fv(gpu_addresses.camera_center, camera_center);

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
        context.uniform1f(gpu_addresses.light_depth_bias, 0.3);
        context.uniform1f(gpu_addresses.light_texture_size, material.lightDepthTextureSize);
        context.uniform1i(gpu_addresses.light_depth_texture, 1);
        if (material.draw_shadow === true) {
            material.light_depth_texture.activate(context, 1);
        }
    }

    shared_glsl_code() {
        return `precision mediump float;
        
                uniform float time;
                
                uniform sampler2D color_texture;
                varying vec2 f_tex_coord;
                
                varying mat3 tbn;
                
                uniform sampler2D specular_texture;
                uniform sampler2D normal_texture;
                uniform bool use_normal;
                
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec3 camera_center;
        
                varying vec3 N, vertex_worldspace;
                vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace, vec2 f_tex_coord, sampler2D specular_texture, sampler2D color_texture, out vec3 light_diffuse_contribution, out vec3 light_specular_contribution ){
                    vec4 specular_tex = texture2D(specular_texture, f_tex_coord);
                    vec4 diffuse_tex = texture2D(color_texture, f_tex_coord);                                        
                    vec3 E = normalize( camera_center - vertex_worldspace );
                    vec3 result = vec3( 0.0 );
                    light_diffuse_contribution = vec3( 0.0 );
                    light_specular_contribution = vec3( 0.0 );
                    for(int i = 0; i < N_LIGHTS; i++){
                        vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                                       light_positions_or_vectors[i].w * vertex_worldspace;                                             
                        float distance_to_light = length( surface_to_light_vector );
        
                        vec3 L = normalize( surface_to_light_vector );
                        vec3 H = normalize( L + E );
                        float diffuse  =      max( dot( N, L ), 0.0 );
                        float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                        float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                        
                        vec3 light_contribution = diffuse_tex.xyz * light_colors[i].xyz * diffusivity * diffuse + specular_tex.xyz * light_colors[i].xyz * specularity * specular;
                        light_diffuse_contribution += attenuation * diffuse_tex.xyz * light_colors[i].xyz * diffusivity * diffuse;
                        light_specular_contribution += attenuation * specular_tex.xyz * specularity * specular;
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
                attribute vec3 tangent;
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                uniform mat4 inv_transpose_model_transform;
                
                void main(){
                    gl_Position = projection_camera_model_transform * vec4(position, 1.0);
                    f_tex_coord = texture_coord;
                    N =  normalize(inv_transpose_model_transform * vec4(normal, 0.0)).xyz;
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                    vec3 BiTangent = cross(normal, tangent);
                    tbn = mat3(normalize(vec3(model_transform * vec4(tangent, 0.0))), normalize(vec3(model_transform * vec4(BiTangent, 0.0))), normalize(vec3(model_transform * vec4(normal, 0.0))));
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
                
                float linearDepth(float val){
                    val = 2.0 * val - 1.0;
                    return (2.0 * 0.5 * 150.0) / (150.0 + 0.5 - val * (150.0 - 0.5));
                }
                
                float PCF_shadow(vec2 center, float projected_depth) {
                    float shadow = 0.0;
                    float texel_size = 1.0 / light_texture_size;
                    for(int x = -1; x <= 1; ++x)
                    {
                        for(int y = -1; y <= 1; ++y)
                        {
                            float light_depth_value = linearDepth(texture2D(light_depth_texture, center + vec2(x, y) * texel_size).r); 
                            shadow += (linearDepth(projected_depth) >= light_depth_value + light_depth_bias) ? 0.8 : 0.0;        
                        }    
                    }
                    shadow /= 9.0;
                    return shadow;
                }
        
                void main(){
                    vec4 color = texture2D(color_texture, f_tex_coord);
                    vec3 normal = texture2D(normal_texture, f_tex_coord).xyz;
                    normal = (normal * 2.0) - 1.0;
                    normal = normalize(tbn * normal);
                    
                    gl_FragColor = vec4(color.xyz * ambient, color.w);
                    
                    vec3 diffuse, specular;
                    if (use_normal){
                        vec3 other_than_ambient = phong_model_lights( normal, vertex_worldspace, f_tex_coord, specular_texture, color_texture, diffuse, specular);
                    }
                    else{
                        vec3 other_than_ambient = phong_model_lights( normalize( N ), vertex_worldspace, f_tex_coord, specular_texture, color_texture, diffuse, specular);
                    }
                    
                    if (draw_shadow) {
                        vec4 light_tex_coord = (light_proj_mat * light_view_mat * vec4(vertex_worldspace, 1.0));
                        light_tex_coord.xyz /= light_tex_coord.w; 
                        light_tex_coord.xyz *= 0.5;
                        light_tex_coord.xyz += 0.5;
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

                }`;
    }
}