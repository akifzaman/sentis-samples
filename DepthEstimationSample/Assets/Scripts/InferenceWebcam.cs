using System.Collections;
using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;

public class InferenceWebcamOptimized : MonoBehaviour
{
    public ModelAsset estimationModel;
    public Texture2D colorMap;
    public RawImage rawImageWebcam;  // Assign in Inspector
    public RawImage rawImageDepth;   // Assign in Inspector

    private Worker engine;
    private WebCamTexture webcamTexture;
    private Tensor<float> inputTensor;
    private RenderTexture resizedWebcamTexture;
    private RenderTexture depthTexture;

    private IEnumerator executionSchedule;
    private bool executionStarted = false;

    private int modelLayerCount = 0;
    private int frameSkip = 3;  // Inference every 3 frames
    public int framesToExecute = 2;

    void Start()
    {
        Application.targetFrameRate = 30;
        QualitySettings.vSyncCount = 0;

        // Load & normalize output of the model
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

        engine = new Worker(model, BackendType.GPUCompute);

        // Start webcam
        webcamTexture = new WebCamTexture(640, 480);
        webcamTexture.Play();
        rawImageWebcam.texture = webcamTexture;

        // Setup render textures
        resizedWebcamTexture = new RenderTexture(256, 256, 0, RenderTextureFormat.ARGB32);
        inputTensor = new Tensor<float>(new TensorShape(1, 3, 256, 256));
        depthTexture = new RenderTexture(256, 256, 0, RenderTextureFormat.ARGBFloat);
        rawImageDepth.texture = depthTexture;
    }

    void Update()
    {
        // Throttle inference
        if (Time.frameCount % frameSkip != 0) return;

        if (!executionStarted)
        {
            // Resize webcam texture to 256x256 using GPU
            Graphics.Blit(webcamTexture, resizedWebcamTexture);
            TextureConverter.ToTensor(resizedWebcamTexture, inputTensor, new TextureTransform().SetDimensions(256, 256));

            executionSchedule = engine.ScheduleIterable(inputTensor);
            executionStarted = true;
        }

        int layersToRun = (modelLayerCount + framesToExecute - 1) / framesToExecute;
        bool hasMoreWork = false;
        for (int i = 0; i < layersToRun; i++)
        {
            hasMoreWork = executionSchedule.MoveNext();
            if (!hasMoreWork) break;
        }

        if (!hasMoreWork)
        {
            var output = engine.PeekOutput() as Tensor<float>;
            output.Reshape(output.shape.Unsqueeze(0));
            TextureConverter.RenderToTexture(output, depthTexture, new TextureTransform().SetCoordOrigin(CoordOrigin.BottomLeft));

            rawImageDepth.texture = depthTexture;
            executionStarted = false;
        }
    }

    void OnDestroy()
    {
        engine?.Dispose();
        inputTensor?.Dispose();
        resizedWebcamTexture?.Release();
        depthTexture?.Release();
    }
}
