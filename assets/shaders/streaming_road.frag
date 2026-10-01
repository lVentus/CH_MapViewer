#version 450 core

uniform vec4 uColor;
uniform uint uRoadStyleDisabled;
uniform uint uRoadStyleConfigValid;
uniform vec4 uRoadColors[64];
layout(location = 0) in float gLodAlpha;
flat in uint gRoadType;
layout(location = 0) out vec4 outColor;

vec3 hueToRgb(float h) {
    h = fract(h);
    float x = h * 6.0;
    int sector = int(floor(x));
    float f = fract(x);
    if (sector == 0) return vec3(1.0, f, 0.0);
    if (sector == 1) return vec3(1.0 - f, 1.0, 0.0);
    if (sector == 2) return vec3(0.0, 1.0, f);
    if (sector == 3) return vec3(0.0, 1.0 - f, 1.0);
    if (sector == 4) return vec3(f, 0.0, 1.0);
    return vec3(1.0, 0.0, 1.0 - f);
}

vec4 fallbackRoadColor(uint roadType) {
    if (roadType == 63u) return vec4(0.70, 0.72, 0.76, 0.95);
    float t = float(min(roadType, 62u)) / 62.0;
    float hue = 0.02 + 0.78 * t;
    vec3 base = hueToRgb(hue);
    float whiten = 0.08 + 0.18 * t;
    vec3 rgb = base * (1.0 - whiten) + vec3(whiten);
    return vec4(rgb, 0.96);
}

void main() {
    uint roadType = min(gRoadType, 63u);
    vec4 style = uRoadStyleDisabled != 0u
                     ? vec4(1.0)
                     : (uRoadStyleConfigValid != 0u
                            ? uRoadColors[roadType]
                            : fallbackRoadColor(roadType));
    outColor = vec4(style.rgb * uColor.rgb,
                    style.a * uColor.a * clamp(gLodAlpha, 0.0, 1.0));
}
