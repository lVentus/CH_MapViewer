#version 430 core

// Color per road type, from the road style config.
uniform vec4 uRoadColors[64];

layout(location = 1) flat in uint gRoadType;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = uRoadColors[min(gRoadType, 63u)];
}
