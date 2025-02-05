using System.Runtime.InteropServices;

namespace Model;

public class NeuralNetwork
{
    private readonly IntPtr _ptr;

    public int Length => GetCount(_ptr);

    public NeuralNetwork(int[] layers)
    {
        _ptr = Initialize(layers.Length, layers);
    }

    [DllImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static extern IntPtr Initialize(int count, int[] layers);

    [DllImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static extern int GetCount(IntPtr ptr);
}