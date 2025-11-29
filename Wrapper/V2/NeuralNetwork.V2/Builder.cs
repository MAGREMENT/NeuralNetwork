using System.Runtime.InteropServices;

namespace NeuralNetwork.V2;

public partial class Builder : IDisposable
{
    private readonly IntPtr _ptr;

    private Builder()
    {
        _ptr = CreateBuilder();
    }

    public static Builder FromYaml(string file)
    {
        var builder = new Builder();
        LoadBuilderFile(builder._ptr, file);
        return builder;
    }

    public NeuralNetwork Build()
    {
        return new NeuralNetwork(Build(_ptr));
    }
    
    public void Dispose()
    {
        DisposeBuilder(_ptr);
    }
    
    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial IntPtr CreateBuilder();
    
    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void DisposeBuilder(IntPtr builder);
    
    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial IntPtr Build(IntPtr builder);
    
    [LibraryImport("Export.dll", StringMarshalling = StringMarshalling.Utf8)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void LoadBuilderFile(IntPtr builder, string file);
}