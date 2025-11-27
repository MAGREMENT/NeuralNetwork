//
// Created by zacha on 26-11-25.
//

#include "training_component.h"

#include <string.h>

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

tc_cnstr_args get_args_from_yaml(yaml_reader* reader, int args_type) {
    switch (args_type) {
        case TCT_INT :
            return TCA_INT(yr_int_seekv(reader, "i", 0));
        case TCT_DOUBLE :
            return TCA_DOUBLE(yr_d_seekv(reader, "d", 0));
        case TCT_DOUBLE_INT :
            return TCA_DOUBLE_INT(yr_d_seekv(reader, "d", 0), yr_int_seekv(reader, "i", 0));
        case TCT_DOUBLE2 :
            return TCA_DOUBLE2(yr_d_seekv(reader, "d1", 0), yr_d_seekv(reader, "d2", 0));
        case TCT_DOUBLE3 :
            return TCA_DOUBLE3(yr_d_seekv(reader, "d1", 0), yr_d_seekv(reader, "d2", 0), yr_d_seekv(reader, "d3", 0));
        default :
            return TCA_NONE;
    }
}

int index_of_tc(const tc_metadata* arr, const int count, const char* str, const int def) {
    for (int i = 0; i < count; i++) {
        if (strcmp(str, arr[i].name) == 0) return i;
    }

    return def;
}