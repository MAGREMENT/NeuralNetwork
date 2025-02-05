namespace Model;

public static class ArrayExtensions
{
    public static T[] Copy<T>(this T[] arr)
    {
        var result = new T[arr.Length];
        arr.CopyTo(result, 0);
        return result;
    }

    public static T[,] To2D<T>(this T[] arr, int width, int height)
    {
        var result = new T[width, height];

        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                result[i, j] = arr[j * width + i];
            }
        }

        return result;
    }
}