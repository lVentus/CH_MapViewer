#version 430 core

out vec4 fragmentColor;

uniform vec4 uColor;

void main() {
    fragmentColor = uColor;
}
