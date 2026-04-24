package com.example.autocropdemo

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.autocropdemo.engine.AutoCropEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var ivOriginal: ImageView
    private lateinit var ivResult: ImageView
    private lateinit var tvStatus: TextView

    private val engine = AutoCropEngine()

    private var sourceBitmap: Bitmap? = null
    private var resultBitmap: Bitmap? = null

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@registerForActivityResult
        val bitmap = decodeBitmap(uri)
        if (bitmap == null) {
            toast("加载图片失败")
            return@registerForActivityResult
        }
        sourceBitmap = bitmap
        resultBitmap = null
        ivOriginal.setImageBitmap(bitmap)
        ivResult.setImageDrawable(null)
        tvStatus.text = "已加载：${bitmap.width}x${bitmap.height}"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ivOriginal = findViewById(R.id.ivOriginal)
        ivResult = findViewById(R.id.ivResult)
        tvStatus = findViewById(R.id.tvStatus)

        val ok = OpenCVLoader.initDebug()
        if (!ok) {
            tvStatus.text = "OpenCV 初始化失败"
            toast("OpenCV 初始化失败")
        } else {
            tvStatus.text = "OpenCV 已就绪，先选择图片"
        }

        findViewById<Button>(R.id.btnPick).setOnClickListener {
            pickImage.launch("image/*")
        }

        findViewById<Button>(R.id.btnAutoCrop).setOnClickListener {
            val src = sourceBitmap
            if (src == null) {
                toast("请先选择图片")
                return@setOnClickListener
            }

            tvStatus.text = "自动裁切中..."
            lifecycleScope.launch {
                val cropResult = withContext(Dispatchers.Default) {
                    engine.autoCrop(src)
                }
                resultBitmap = cropResult.bitmap
                ivResult.setImageBitmap(cropResult.bitmap)
                tvStatus.text = "完成：${cropResult.debugInfo}"
            }
        }

        findViewById<Button>(R.id.btnSave).setOnClickListener {
            val bmp = resultBitmap
            if (bmp == null) {
                toast("请先执行自动裁切")
                return@setOnClickListener
            }
            val uri = savePngToGallery(bmp)
            if (uri != null) {
                toast("已保存到相册")
                tvStatus.text = "保存成功：$uri"
            } else {
                toast("保存失败")
            }
        }
    }

    private fun decodeBitmap(uri: Uri): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                    decoder.isMutableRequired = true
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun savePngToGallery(bitmap: Bitmap): Uri? {
        val fileName = "autocrop_${timestamp()}.png"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/AutoCropDemo")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }

        val collection = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        val itemUri = contentResolver.insert(collection, values) ?: return null

        return try {
            val out: OutputStream = contentResolver.openOutputStream(itemUri) ?: return null
            out.use {
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                contentResolver.update(itemUri, values, null, null)
            }
            itemUri
        } catch (e: Exception) {
            contentResolver.delete(itemUri, null, null)
            null
        }
    }

    private fun timestamp(): String {
        val sdf = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
        return sdf.format(Date())
    }

    private fun toast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }
}
