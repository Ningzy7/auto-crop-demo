package com.example.autocropdemo.engine

import android.graphics.Bitmap
import android.graphics.Rect

data class CropResult(
    val bitmap: Bitmap,
    val rect: Rect,
    val debugInfo: String = ""
)
