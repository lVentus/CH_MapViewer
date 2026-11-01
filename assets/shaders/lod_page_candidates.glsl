// Output of the filter kernels: the candidate list, appended once per workgroup.

layout(std430, binding = 3) buffer Candidates {
    uint candidateCount;
    uint candidateGroups[3];
    uint candidates[];
};

uniform uint uCandidateCapacity;

shared uint sKeptCount;
shared uint sKeptBase;

// Called by every invocation of the workgroup.
void keepCandidates(bool keep, uint item) {
    if (gl_LocalInvocationIndex == 0u) {
        sKeptCount = 0u;
    }
    barrier();
    uint offset = 0u;
    if (keep) {
        offset = atomicAdd(sKeptCount, 1u);
    }
    barrier();
    if (gl_LocalInvocationIndex == 0u && sKeptCount != 0u) {
        sKeptBase = atomicAdd(candidateCount, sKeptCount);
    }
    barrier();
    if (keep) {
        uint index = sKeptBase + offset;
        if (index < uCandidateCapacity) {
            candidates[index] = item;
        } else {
            atomicAdd(status[kStatusCandidateOverflow], 1u);
        }
    }
}
