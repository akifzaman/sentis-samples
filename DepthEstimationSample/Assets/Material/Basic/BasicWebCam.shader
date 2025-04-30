Shader "Unlit/BasicWebCam"
{
    Properties
    {
        _MainTex ("Base Texture", 2D) = "white" { }
    }
    SubShader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct v2f
            {
                float4 pos : POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(float4 vertex : POSITION)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(vertex);
                o.uv = vertex.xy * 0.5 + 0.5; // UV mapping for full-screen quad
                return o;
            }

            sampler2D _MainTex;
            float4 frag(v2f i) : SV_Target
            {
                return tex2D(_MainTex, i.uv);  // Just show the webcam texture
            }
            ENDCG
        }
    }
}
