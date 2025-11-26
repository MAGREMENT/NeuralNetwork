//
// Created by zacha on 26-11-25.
//

#include "training_component.h"

void add_to_yaml_writer(yaml_writer* writer, tc_cnstr_args args, int args_type) {
    if (args_type == TCT_NONE) return;

    yw_begin_map(writer);

    switch (args_type) {
        case TCT_INT :
            yw_int_nv(writer, "i", args.i);
            break;
        case TCT_DOUBLE :
            yw_d_nv(writer, "d", args.d);
            break;
        case TCT_DOUBLE_INT :
            yw_d_nv(writer, "d", args.di.d);
            yw_int_nv(writer, "i", args.di.i);
            break;
        case TCT_DOUBLE2 :
            yw_d_nv(writer, "d1", args.d2.d1);
            yw_d_nv(writer, "d2", args.d2.d2);
            break;
        case TCT_DOUBLE3 :
            yw_d_nv(writer, "d1", args.d3.d1);
            yw_d_nv(writer, "d2", args.d3.d2);
            yw_d_nv(writer, "d3", args.d3.d3);
            break;
        default : break;
    }

    yw_end(writer);
}