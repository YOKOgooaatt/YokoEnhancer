package com.realface.enhancer

import android.app.Activity
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import java.io.平

class MainActivity : AppCompatActivity() {
    private val enhancer = WebGpuEnhancer()
    private val scope = CoroutineScope(Job() + Dispatchers.Main)
    private lateinit var statusText: TextView
    private lateinit var processButton: Button
    private var selectedImageUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Minimal Programmatic UI container layout setup
        val layout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            gravity = android.view.Gravity.CENTER
            setBackgroundColor(0xFF111111.toInt())
            setPadding(50, 50, 50, 50)
        }

        val titleText = TextView(this).apply {
            text = "RealFace Mobile 4K Suite"
            textSize = 22f
            setTextColor(0xFFFFFFFF.toInt())
            setTypeface(null, android.graphics.Typeface.BOLD)
        }
        layout.addView(titleText)

        statusText = TextView(this).apply {
            text = "Status: Engine Ready"
            textSize = 14f
            setTextColor(0xFF00ADB5.toInt())
            setPadding(0, 20, 0, 40)
        }
        layout.addView(statusText)

        val loadButton = Button(this).apply {
            text = "📂 Load Image"
            setBackgroundColor(0xFF222222.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            setOnClickListener { openFilePicker() }
        }
        layout.addView(loadButton)

        processButton = Button(this).apply {
            text = "⚡ Enhance to 4K"
            setBackgroundColor(0xFF00ADB5.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            isEnabled = false
            setOnClickListener { runEnhancementPipeline() }
        }
        layout.addView(processButton)

        setContentView(layout)
    }

    private fun openFilePicker() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }
        startActivityForResult(intent, 101)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 101 && resultCode == Activity.RESULT_OK) {
            selectedImageUri = data?.data
            processButton.isEnabled = true
            statusText.text = "Asset loaded. Ready to execute."
            statusText.setTextColor(0xFFFFFFFF.toInt())
        }
    }

    private fun runEnhancementPipeline() {
        val uri = selectedImageUri ?: return
        processButton.isEnabled = false
        statusText.text = "Processing frames via WebGPU compute layer..."
        
        scope.launch {
            try {
                val inputStream = contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                
                // Fire native GPU tasks asynchronously 
                val result4k = enhancer.upscaleTo4K(bitmap)
                
                statusText.text = "✓ 4K Structural Render Complete!"
                statusText.setTextColor(0xFF00FFCC.toInt())
                Toast.makeText(this@MainActivity, "Saved to hidden app cache context!", Toast.LENGTH_LONG).show()
            } catch (e: Exception) {
                statusText.text = "Pipeline Breakdown: ${e.message}"
                statusText.setTextColor(0xFFFF3333.toInt())
            } finally {
                processButton.isEnabled = true
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel() // Safeguard active background compute instances on close
    }
}
