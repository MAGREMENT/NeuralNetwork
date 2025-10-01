using System.Runtime.InteropServices;

namespace Model;

public partial class LearningState : IDisposable
{
    private readonly int _layerCount;
    
    internal IntPtr Ptr { get; }

    public LearningState(NeuralNetwork network)
    {
        Ptr = CreateState(network.Ptr);
        _layerCount = network.Length;
    }

    public void Dispose()
    {
        DisposeState(Ptr, _layerCount);
    }
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial IntPtr CreateState(IntPtr ptr);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void DisposeState(IntPtr ptr, int layerCount);
}