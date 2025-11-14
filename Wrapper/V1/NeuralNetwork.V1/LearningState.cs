using System.Runtime.InteropServices;

namespace NeuralNetwork.V1;

public partial class LearningState : IDisposable
{
    private readonly NeuralNetwork _network;
    
    internal IntPtr Ptr { get; }

    public LearningState(NeuralNetwork network)
    {
        Ptr = CreateState(network.Ptr);
        _network = network;
    }

    public void Dispose()
    {
        DisposeState(_network.Ptr, Ptr);
    }
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial IntPtr CreateState(IntPtr ptr);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void DisposeState(IntPtr ptr, IntPtr state);
}