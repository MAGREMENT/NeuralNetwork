using System.Runtime.InteropServices;
using Base;

namespace NeuralNetwork.V2;

public partial class LearningState : ILearningState
{
    private readonly IntPtr _ptr;
    private readonly NeuralNetwork _network;

    public LearningState(NeuralNetwork network)
    {
        _ptr = CreateState(network.GetPointer());
        _network = network;
    }

    public void Dispose()
    {
        DisposeState(_network.GetPointer(), _ptr);
    }

    public IntPtr GetPointer() => _ptr;
    
    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial IntPtr CreateState(IntPtr ptr);

    [LibraryImport("Export.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void DisposeState(IntPtr ptr, IntPtr state);
}