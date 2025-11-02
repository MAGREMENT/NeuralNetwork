//
// Created by zacha on 28-10-25.
//

#ifndef NEWIMPLEMENTATION_SIZE_H
#define NEWIMPLEMENTATION_SIZE_H

typedef struct size3D {
    int width;
    int height;
    int depth;
} size3D;

typedef struct size2D {
    int width;
    int height;
} size2D;

size2D to2D(size3D s);
size3D to3D(size2D s, int depth);

#endif //NEWIMPLEMENTATION_SIZE_H