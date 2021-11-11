import {defs, tiny} from './examples/common.js';
const {vec3, vec4, vec, color, Matrix, Mat4, Light, Shape, Material, Shader, Texture, Scene} = tiny;

//im not gonna comment on how the shaders work for now. If you want to know how they work, ask me in person and ill go through them
//This shader creates the skybox by blending between 3 values, one for the horizon, under the horizon, and above the horizon
export class Skybox_Shader extends Shader {

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

//the grass shader for the terrain. Has built in Phong lighting. To work correctly it needs to have its layer property set
//and incremented in successive draw calls. I find that drawing with 16 layers looks good enough
//doesn't draw grass when the vertex's world position is under a certain threshold (so no grass under water)
//no grass will be drawn where the occlusion texture's red value is not 0
export class Grass_Shader extends Shader {
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

export class Grass_Shader_Shadow extends Shader {
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
        context.uniform1i(gpu_addresses.ground_texture, 4);
        material.ground_texture.activate(context, 4);

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
        context.uniform1f(gpu_addresses.light_depth_bias, 0.5);
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
                uniform sampler2D ground_texture;
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
                            vec4 groundTexColor = texture2D(ground_texture, f_tex_coord * 10.0);
                            light_contribution = groundTexColor.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                            light_diffuse_contribution += attenuation * groundTexColor.xyz * light_colors[i].xyz * diffusivity * diffuse;
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
                
                void main(){
                    worldPos = model_transform * vec4(position, 1.0);
                    float alpha = (layer / 10.0) * PerlinNoise3Pass(worldPos.xz + vec2(time * 1.5, time * 1.5), 2.0);
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
                
                float linearDepth(float val){
                    val = 2.0 * val - 1.0;
                    return (2.0 * 0.5 * 500.0) / (500.0 + 0.5 - val * (500.0 - 0.5));
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
                    vec4 groundTexColor = texture2D(ground_texture, f_tex_coord * 60.0);
                    if (fake_shadow_layer){
                        float perlin = 1.0 - (1.0 - PerlinNoise3Pass(vec2(worldPos.x + 0.02, worldPos.z - 0.02), 50.0)) * 2.4;
                        float white = 1.0 - (1.0 - perlinNoise(vec2(worldPos.x + 0.02, worldPos.z - 0.02))) * 40.0;
                        float alpha = perlin * white - ((layer + 0.2) * 2.2 / 1.0);
                        if (alpha < 0.0 || worldPos.y < -1.0){
                            discard;
                        }
                        gl_FragColor = vec4(grass_color.x * ambient * 2.35 + (layer / 200.0), grass_color.y * ambient * 2.35 + (layer / 200.0), grass_color.z * ambient * 2.35 + (layer / 200.0), 1.0);
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
                        gl_FragColor = vec4(groundTexColor.x * ambient, groundTexColor.y * ambient, groundTexColor.z * ambient, 1.0);
                    }
                    vec3 diffuse, specular;
                    vec3 other_than_ambient = phong_model_lights( normalize( N ), vertex_worldspace, diffuse, specular );
                    
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
                    
                    if (layer > 0.0){
                        float perlin = 1.0 - (1.0 - PerlinNoise3Pass(worldPos.xz, 50.0)) * 2.2;
                        float white = 1.0 - (1.0 - perlinNoise(worldPos.xz)) * 40.0;
                        float alpha = perlin * white - ((layer + 0.2) * 1.2 / 1.0);
                        if (alpha < 0.0 || worldPos.y < -1.0){
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

export class Grass_Shader_Shadow_Textured extends Shader {
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
        material.texture.activate(context, 0);
        context.uniform1i(gpu_addresses.noiseTexture1, 3);
        material.noiseTexture1.activate(context, 3);
        context.uniform1i(gpu_addresses.noiseTexture2, 2);
        material.noiseTexture2.activate(context, 2);

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
        context.uniform1f(gpu_addresses.light_depth_bias, 0.0015);
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
                uniform sampler2D noiseTexture1;
                uniform sampler2D noiseTexture2;
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
                    //float alpha = (layer / 10.0) * PerlinNoise3Pass(worldPos.xz + vec2(time * 2.0, time * 2.0), 2.0);
                    //gl_Position = projection_camera_model_transform * vec4(position.x + (0.2 * alpha), position.y + (0.04 * layer), position.z + (0.2 * alpha), 1.0);
                    gl_Position = projection_camera_model_transform * vec4(position.x, position.y + (0.04 * layer), position.z, 1.0);
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
                
                float linearDepth(float val){
                    val = 2.0 * val - 1.0;
                    return (2.0 * 0.5 * 500.0) / (500.0 + 0.5 - val * (500.0 - 0.5));
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
                        vec4 noise1 = texture2D(noiseTexture1, f_tex_coord / 1.8);
                        vec4 noise2 = texture2D(noiseTexture2, f_tex_coord * 3.0);
                        float alpha = ((noise1.x * 2.3) * pow((noise2.x * 3.0), 3.5)) - ((layer + 5.5) * 1.2 / 1.0);
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

export class Grass_Shader_Background_Shadow extends Shader {
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
        context.uniform1i(gpu_addresses.fake_shadow_layer, material.fake_shadow_layer);
        context.uniform1i(gpu_addresses.ground_texture, 4);
        material.ground_texture.activate(context, 4);

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
        context.uniform1f(gpu_addresses.light_depth_bias, 0.3);
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
                
                uniform sampler2D ground_texture;
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
                            light_diffuse_contribution += attenuation * groundTexColor.xyz * light_colors[i].xyz * diffusivity * diffuse;
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
                attribute vec3 normal;
                attribute vec2 texture_coord;
                uniform mat4 projection_camera_model_transform;
                uniform mat4 model_transform;
                
                void main(){
                    worldPos = model_transform * vec4(position, 1.0);
                    float alpha = (layer / 10.0) * PerlinNoise3Pass(worldPos.xz + vec2(time * 1.5, time * 1.5), 2.0);
                    gl_Position = projection_camera_model_transform * vec4(position.x + (0.02 * alpha), position.y + (0.04 * layer), position.z + (0.02 * alpha), 1.0);
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                    f_tex_coord = texture_coord;
                    
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
                
                float linearDepth(float val){
                    val = 2.0 * val - 1.0;
                    return (2.0 * 0.5 * 500.0) / (500.0 + 0.5 - val * (500.0 - 0.5));
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
                    vec4 groundTexColor = texture2D(ground_texture, f_tex_coord * 60.0);
                    if (fake_shadow_layer){
                        float perlin = 1.0 - (1.0 - PerlinNoise3Pass(vec2(worldPos.x + 0.02, worldPos.z - 0.02), 50.0)) * 2.4;
                        float white = 1.0 - (1.0 - perlinNoise(vec2(worldPos.x + 0.02, worldPos.z - 0.02))) * 40.0;
                        float alpha = perlin * white - ((layer + 0.2) * 2.2 / 1.0);
                        if (alpha < 0.0 || worldPos.y < -0.7){
                            discard;
                        }
                        gl_FragColor = gl_FragColor = vec4(grass_color.x * ambient * 2.35 + (layer / 200.0), grass_color.y * ambient * 2.35 + (layer / 200.0), grass_color.z * ambient * 2.35 + (layer / 200.0), 1.0 - exp(-0.5 * (40.0 - distance(vec4(0,0,0,0), worldPos))));
                    }
                    else{
                    if (layer > 0.0){
                        gl_FragColor = vec4(grass_color.x * ambient + (layer / 70.0), grass_color.y * ambient + (layer / 70.0), grass_color.z * ambient + (layer / 70.0), 1.0 - exp(-0.5 * (40.0 - distance(vec4(0,0,0,0), worldPos))));
                    }
                    else {
                        gl_FragColor = vec4(groundTexColor.x * ambient + (layer / 70.0), groundTexColor.y * ambient + (layer / 70.0), groundTexColor.z * ambient + (layer / 70.0), 1.0 - exp(-0.5 * (40.0 - distance(vec4(0,0,0,0), worldPos))));
                    }
                    vec3 diffuse, specular;
                    vec3 other_than_ambient = phong_model_lights( normalize( N ), vertex_worldspace, diffuse, specular );
                    
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

                    if (layer > 0.0){
                        float perlin = 1.0 - (1.0 - PerlinNoise3Pass(worldPos.xz, 50.0)) * 2.2;
                        float white = 1.0 - (1.0 - perlinNoise(worldPos.xz)) * 40.0;
                        float alpha = perlin * white - ((layer + 0.2) * 1.2 / 1.0);
                        if (alpha < 0.0 || worldPos.y < -1.0){
                            discard;
                        }
                    }
                    }
                }`;
    }
}

//work in progress. I want to use a texture instead of procedural noise for the performance
export class Grass_Shader_Background_Textured extends Shader {
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
export class Water_Shader extends Shader{

    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000);
        context.uniform4fv(gpu_addresses.shallow_color, material.shallow_color);
        context.uniform4fv(gpu_addresses.deep_color, material.deep_color);
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

        context.uniform1i(gpu_addresses.depth_texture, 1);
        material.depth_texture.activate(context, 1);

        //context.uniform1i(gpu_addresses.bg_color_texture, 3);
        //material.bg_color_texture.activate(context, 3);
    }

    shared_glsl_code() {
        return `precision mediump float;
                uniform float time;
                
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec3 squared_scale, camera_center;
                uniform vec4 shallow_color;
                uniform vec4 deep_color;
        
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
                        
                        vec3 light_contribution = shallow_color.xyz * light_colors[i].xyz * diffusivity * diffuse
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
                
                vec3 GerstnerWave (vec4 wave, vec3 p, inout vec3 tangent, inout vec3 binormal) {
                    float steepness = wave.z;
                    float wavelength = wave.w;
                    float k = 2.0 * 3.141592653589 / wavelength;
                    float c = sqrt(9.8 / k);
                    vec2 d = normalize(wave.xy);
                    float f = k * (dot(d, p.xz) - c * time);
                    float a = steepness / k;

                    tangent += vec3(
                       -d.x * d.x * (steepness * sin(f)),
                        d.x * (steepness * cos(f)),
                       -d.x * d.y * (steepness * sin(f)));
                    binormal += vec3(
                        -d.x * d.y * (steepness * sin(f)),
                        d.y * (steepness * cos(f)),
                        -d.y * d.y * (steepness * sin(f)));
                    return vec3(
                        d.x * (a * cos(f)),
                        a * sin(f),
                        d.y * (a * cos(f)));
                }
                
                void main(){
                    gl_Position = projection_camera_model_transform * vec4( position.x, position.y, position.z, 1.0 );
                    N = normalize( mat3( model_transform ) * vec3(normal.x, normal.y, normal.z) / squared_scale);
                    vertex_worldspace = (model_transform * vec4( position, 1.0 )).xyz;
                }`;
    }

    //sets each pixel's color
    fragment_glsl_code() {
        return this.shared_glsl_code() + `
                uniform sampler2D depth_texture;
                //uniform sampler2D bg_color_texture;
                
                float linearDepth(float val){
                    val = 2.0 * val - 1.0;
                    return (2.0 * 1.0 * 500.0) / (500.0 + 1.0 - val * (500.0 - 1.0));
                }
                
                void main(){
                    float depthVal = texture2D(depth_texture, vec2((gl_FragCoord.x - 0.5) / 1919.0, (gl_FragCoord.y - 0.5) / 1079.0)).r;
                    //vec4 bgColor = texture2D(bg_color_texture, vec2((gl_FragCoord.x - 0.5) / 1919.0, (gl_FragCoord.y - 0.5) / 1079.0));
                    float depthDifference = abs(linearDepth(depthVal) - linearDepth(gl_FragCoord.z));
                    gl_FragColor = vec4(mix(shallow_color.xyz, deep_color.xyz, sin(min(depthDifference / 5.0, 1.0) * 3.14159 / 2.0 )), sin(min(depthDifference / 2.0, 1.0) * 3.14159 / 2.0 ));
                    //gl_FragColor = bgColor;
                    
                    vec3 lighting = phong_model_lights( normalize( N ), vertex_worldspace);
                    gl_FragColor.xyz += lighting;
                }`;
    }
}

export class Shadow_Textured_Phong extends Shader {
    constructor(layer, num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000.0);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniform1i(gpu_addresses.color_texture, 0);
        material.color_texture.activate(context, 0);

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
        context.uniform1f(gpu_addresses.light_depth_bias, 0.3);
        context.uniform1f(gpu_addresses.light_texture_size, material.lightDepthTextureSize);
        context.uniform1i(gpu_addresses.light_depth_texture, 1);
        if (material.draw_shadow) {
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
                uniform vec3 squared_scale, camera_center;
        
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
                
                void main(){
                    gl_Position = projection_camera_model_transform * vec4(position, 1.0);
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
                
                float linearDepth(float val){
                    val = 2.0 * val - 1.0;
                    return (2.0 * 0.5 * 500.0) / (500.0 + 0.5 - val * (500.0 - 0.5));
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

export class Shadow_Textured_Phong_Maps extends Shader {
    constructor(layer, num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform], PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

        context.uniform1f(gpu_addresses.time, graphics_state.animation_time / 1000.0);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
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
        context.uniform1f(gpu_addresses.light_depth_bias, 0.3);
        context.uniform1f(gpu_addresses.light_texture_size, material.lightDepthTextureSize);
        context.uniform1i(gpu_addresses.light_depth_texture, 1);
        if (material.draw_shadow) {
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
                uniform vec3 squared_scale, camera_center;
        
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
                
                void main(){
                    gl_Position = projection_camera_model_transform * vec4(position, 1.0);
                    f_tex_coord = texture_coord;
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
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
                    return (2.0 * 0.5 * 500.0) / (500.0 + 0.5 - val * (500.0 - 0.5));
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