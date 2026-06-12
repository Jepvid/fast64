"""
smooth_skin.py — HM64 smooth skinning export.

Collects vertices influenced by 2+ bones for each skeleton limb and serialises
them into a <SmoothSkin> XML block that gets embedded inside the <SkeletonLimb>
element produced by _OOTLimb_toSohXML.

Public API
----------
build_smooth_skin_data(meshObj, armatureObj, boneName, convertTransformMatrix, limbList)
    -> dict[pos_key, list[influence]] | None

smooth_skin_to_xml(smooth_data) -> str
"""

import mathutils

WEIGHT_THRESHOLD = 0.001
MAX_INFLUENCES = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bone_local_transform(armatureObj, boneName, convertTransformMatrix):
    """Returns the 4x4 matrix that maps a vertex from armature-local space into bone-local game space."""
    bone = armatureObj.data.bones.get(boneName)
    if bone is None:
        return convertTransformMatrix.copy()
    return convertTransformMatrix @ bone.matrix_local.inverted()


def _mesh_to_armature_matrix(meshObj, armatureObj):
    """Returns the 4x4 matrix that converts mesh-local vertex coords into armature-local coords."""
    return armatureObj.matrix_world.inverted() @ meshObj.matrix_world


def _bone_name_to_limb_index(boneName, limbList):
    for limb in limbList:
        if limb.boneName == boneName:
            return limb.index
    return -1


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def build_smooth_skin_data(meshObj, armatureObj, boneName, convertTransformMatrix, limbList):
    """
    For all vertices whose DOMINANT bone is `boneName` and that are weighted
    to 2+ bones, return a dict:

        { (domX, domY, domZ): [ {bone, weight, localX, localY, localZ, normX, normY, normZ}, ... ] }

    The dom* key is the vertex position in the dominant bone's local game space (integer-rounded).

    Returns None when no multi-influence vertices exist for this limb.
    """
    mesh = meshObj.data
    bones = armatureObj.data.bones

    # Group index for the dominant bone of this limb
    domGroupIndex = None
    for group in meshObj.vertex_groups:
        if group.name == boneName:
            domGroupIndex = group.index
            break

    if domGroupIndex is None:
        return None

    # Cache group index → bone name for deform bones only
    groupIndexToBone = {
        group.index: group.name
        for group in meshObj.vertex_groups
        if group.name in bones
    }

    meshToArm = _mesh_to_armature_matrix(meshObj, armatureObj)
    domTransform = _bone_local_transform(armatureObj, boneName, convertTransformMatrix)

    result = {}

    for vert in mesh.vertices:
        # Collect valid bone influences for this vertex
        validInfluences = [
            g for g in vert.groups
            if g.weight > WEIGHT_THRESHOLD and g.group in groupIndexToBone
        ]

        if not validInfluences:
            continue

        # Dominant bone = highest weight
        dominant = max(validInfluences, key=lambda g: g.weight)
        if dominant.group != domGroupIndex:
            continue  # another bone owns this vertex

        # Skip single-influence vertices
        if len(validInfluences) < 2:
            continue

        # Vertex position in armature-local space
        armVert = meshToArm @ vert.co.to_4d()

        # Position in dominant bone's game space
        domPos = domTransform @ armVert
        posKey = (round(domPos.x), round(domPos.y), round(domPos.z))

        if posKey in result:
            continue  # duplicate position, skip

        # Normalise weights (cap to MAX_INFLUENCES, keep heaviest)
        sorted_inf = sorted(validInfluences, key=lambda g: g.weight, reverse=True)[:MAX_INFLUENCES]
        totalWeight = sum(g.weight for g in sorted_inf)
        if totalWeight < WEIGHT_THRESHOLD:
            continue

        # Vertex normal in armature-local space
        armNorm = meshToArm.to_3x3() @ vert.normal

        influences = []
        weightAccum = 0

        for i, g in enumerate(sorted_inf):
            boneName_i = groupIndexToBone[g.group]
            limbIndex = _bone_name_to_limb_index(boneName_i, limbList)
            if limbIndex < 0:
                continue

            # Normalised weight 0-255; last influence absorbs any rounding remainder
            if i < len(sorted_inf) - 1:
                w = round((g.weight / totalWeight) * 255)
            else:
                w = 255 - weightAccum
            weightAccum += w

            boneTransform = _bone_local_transform(armatureObj, boneName_i, convertTransformMatrix)

            localPos = boneTransform @ armVert
            localNorm = (boneTransform.to_3x3() @ armNorm).normalized()

            influences.append({
                "bone":   limbIndex,
                "weight": w,
                "localX": round(localPos.x),
                "localY": round(localPos.y),
                "localZ": round(localPos.z),
                "normX":  max(-128, min(127, round(localNorm.x * 127))),
                "normY":  max(-128, min(127, round(localNorm.y * 127))),
                "normZ":  max(-128, min(127, round(localNorm.z * 127))),
            })

        if len(influences) >= 2:
            result[posKey] = influences

    return result if result else None


# ---------------------------------------------------------------------------
# XML serialisation
# ---------------------------------------------------------------------------

def smooth_skin_to_xml(smooth_data, indent="\t"):
    """
    Converts the dict returned by build_smooth_skin_data into a <SmoothSkin>
    XML string, ready to be embedded inside a <SkeletonLimb> element.
    """
    i1 = indent
    i2 = indent * 2
    i3 = indent * 3

    lines = [f"{i1}<SmoothSkin>"]

    for (domX, domY, domZ), influences in smooth_data.items():
        lines.append(f'{i2}<Vertex DomX="{domX}" DomY="{domY}" DomZ="{domZ}">')
        for inf in influences:
            lines.append(
                f'{i3}<Influence'
                f' Bone="{inf["bone"]}"'
                f' Weight="{inf["weight"]}"'
                f' LocalX="{inf["localX"]}"'
                f' LocalY="{inf["localY"]}"'
                f' LocalZ="{inf["localZ"]}"'
                f' NormX="{inf["normX"]}"'
                f' NormY="{inf["normY"]}"'
                f' NormZ="{inf["normZ"]}"/>'
            )
        lines.append(f"{i2}</Vertex>")

    lines.append(f"{i1}</SmoothSkin>")
    return "\n".join(lines) + "\n"
