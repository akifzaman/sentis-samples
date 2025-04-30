using System.Collections;
using UnityEngine;
using Unity.Sentis;

public class DepthInferenceCamera : MonoBehaviour
{
    public ModelAsset estimationModel;
    public Material material;
    public Texture2D colorMap;
    public Camera captureCamera;

    Worker m_engineEstimation;
    Tensor<float> inputTensor;
    RenderTexture inputRT;     // Small RT for inference
    RenderTexture outputRT;    // Depth result
    RenderTexture cameraViewRT; // Fullscreen RT for shader display
    IEnumerator executionSchedule;

    int modelLayerCount;
    bool executionStarted;
    public int framesToExecute = 6;
    public int inferenceFrameSkip = 2;

    void Start()
    {
        Application.targetFrameRate = 30;

        // Load and normalize model
        var model = ModelLoader.Load(estimationModel);
        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model);
        var outputs = Functional.Forward(model, inputs);
        var output = outputs[0];
        var max0 = Functional.ReduceMax(output, new[] { 1, 2 }, false);
        var min0 = Functional.ReduceMin(output, new[] { 1, 2 }, false);
        output = (output - min0) / (max0 - min0);
        model = graph.Compile(output);

        modelLayerCount = model.layers.Count;
        m_engineEstimation = new Worker(model, BackendType.GPUCompute);

        // Small RenderTexture for inference
        inputRT = new RenderTexture(128, 128, 0, RenderTextureFormat.ARGB32);
        inputRT.Create();

        // Large RT for the visual output
        cameraViewRT = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.ARGB32);
        cameraViewRT.Create();

        // Depth output
        outputRT = new RenderTexture(128, 128, 0, RenderTextureFormat.ARGBFloat);
        outputRT.Create();

        inputTensor = new Tensor<float>(new TensorShape(1, 3, 128, 128));

        // Shader constants
        material.SetTexture("ColorRampTex", colorMap);

        // Camera renders to the full display RT
        captureCamera.targetTexture = cameraViewRT;
    }

    void Update()
    {
        if (Time.frameCount % inferenceFrameSkip != 0)
            return;

        // Blit downscaled version for inference
        Graphics.Blit(cameraViewRT, inputRT);

        if (!executionStarted)
        {
            TextureConverter.ToTensor(inputRT, inputTensor, new TextureTransform());
            executionSchedule = m_engineEstimation.ScheduleIterable(inputTensor);
            executionStarted = true;
        }

        int layersPerFrame = Mathf.CeilToInt((float)modelLayerCount / framesToExecute);
        bool hasMoreWork = false;

        for (int i = 0; i < layersPerFrame; i++)
        {
            hasMoreWork = executionSchedule.MoveNext();
            if (!hasMoreWork)
                break;
        }

        if (hasMoreWork)
            return;

        var output = m_engineEstimation.PeekOutput() as Tensor<float>;
        output.Reshape(output.shape.Unsqueeze(0));
        TextureConverter.RenderToTexture(output, outputRT, new TextureTransform().SetCoordOrigin(CoordOrigin.BottomLeft));
        executionStarted = false;
    }

    void OnRenderObject()
    {
        material.SetVector("ScreenCamResolution", new Vector4(Screen.height, Screen.width, 0, 0));
        material.SetTexture("WebCamTex", cameraViewRT); // Full visual camera feed
        material.SetTexture("DepthTex", outputRT);       // Depth result
        Graphics.Blit(null, (RenderTexture)null, material); // Fullscreen display
    }

    void OnDestroy()
    {
        m_engineEstimation?.Dispose();
        inputTensor?.Dispose();
        inputRT?.Release();
        outputRT?.Release();
        cameraViewRT?.Release();
    }
}
