//
// Created by zacha on 02-11-25.
//

#include "size.h"

inline size2D to2D(const size3D s) {
    return (size2D) {s.width, s.height};
}

inline size3D to3D(const size2D s, const int depth) {
    return (size3D) {s.width, s.height, depth};
}
