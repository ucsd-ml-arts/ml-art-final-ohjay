import os
import sys
import bpy
from bpy.props import StringProperty

class ImportOBJ(bpy.types.Operator):
    bl_idname = 'import_scene.import_obj'
    bl_label = 'Import OBJ'

    # Properties
    in_filepath = StringProperty(name='Input Filepath')
    
    def execute(self, context):
        # Delete default cube
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Cube'].select = True
        bpy.ops.object.delete() 

        bpy.ops.import_scene.obj(filepath=self.in_filepath)
        return {'FINISHED'}

class ExportOBJ(bpy.types.Operator):
    bl_idname = 'export_scene.export_obj'
    bl_label = 'Export OBJ'

    # Properties
    out_filepath = StringProperty(name='Output Filepath')

    def execute(self, context):
        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(self.out_filepath)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print('Created directory at %s' % out_dir)

        # UV map mesh
        scene = bpy.context.scene
        for obj in scene.objects:
            if obj.type == 'MESH':
                obj.select = True
                scene.objects.active = obj
                bpy.ops.object.editmode_toggle()
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.uv.unwrap()

        # for mesh in [obj.data for obj in bpy.context.scene.objects if obj.type == 'MESH']:
        #     print(mesh)
        #     mesh.uv_textures.new('uv')

        bpy.ops.export_scene.obj(filepath=self.out_filepath)
        print('Exported OBJ to %s.\n\n' % self.out_filepath)
        return {'FINISHED'}

def register():
    bpy.utils.register_class(ImportOBJ)
    bpy.utils.register_class(ExportOBJ)

def unregister():
    bpy.utils.unregister_class(ImportOBJ)
    bpy.utils.unregister_class(ExportOBJ)

if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--')+1:]
    assert len(argv) >= 2, 'must provide an input and output path'
    in_filepath = argv[0]
    print('input filepath: %s' % in_filepath)
    out_filepath = argv[1]
    print('output filepath: %s' % out_filepath)

    register()
    bpy.ops.import_scene.import_obj(in_filepath=in_filepath)
    bpy.ops.export_scene.export_obj(out_filepath=out_filepath)
