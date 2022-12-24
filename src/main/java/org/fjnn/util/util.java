/*
 * The MIT License
 *
 * Copyright 2018 Ahmed Tarek.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

package org.fjnn.util;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.zip.DataFormatException;
import java.util.zip.Deflater;
import java.util.zip.Inflater;

/**
 *
 * @author Ahmed Tarek
 */
public class util {    
        
    public static byte[] readFile(String filename) throws IOException {
        try (FileInputStream fs = new FileInputStream(filename)) {
            byte[] result = new byte[fs.available()];
            fs.read(result);
            fs.close();
            return result;
        }
    }

    public static float[] copyArray(float[] array) {
        float[] copy = new float[array.length];

        System.arraycopy(array, 0, copy, 0, array.length);
        
        return copy;
    }
    
    public static float[][] copyArray(float[][] array) {
        float[][] copy = new float[array.length][];

        for(int i=0; i < array.length; i++) {
            copy[i] = new float[array[i].length];
            System.arraycopy(array[i], 0, copy[i], 0, array[i].length);
        }
        
        return copy;
    }
    
    public static float[] copyArray(float[] array, int length) {
        float[] copy = new float[length];

        System.arraycopy(array, 0, copy, 0, array.length);
        
        return copy;
    }
    
    public static void addArray(float[] accumulator, float[] array) {
        if(accumulator.length != array.length)
            throw new RuntimeException("invalid array length: " + accumulator.length + " " + array.length);
        
        for(int i=0; i < accumulator.length; i++)
            accumulator[i] += array[i];
    }
    
    public static void printArray(float[] array) {
        printArray(array, 0, array.length);
    }
    
    public static void printArray(float[] array, int start, int length) {
        for(int i=start; i < start+length; i++)
            System.out.print(array[i] + " ");
        System.out.println();
    }
    
    public static String readStream(InputStream stream) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte buffer[] = new byte[4096];
        
        int len;
        
        while((len = stream.read(buffer)) != -1)
            output.write(buffer, 0, len);

        return output.toString("UTF-8");
    }

    public static void saveFile(String filename, byte[] data) throws IOException {
        try (FileOutputStream fs = new FileOutputStream(filename)) {
            fs.write(data);
            fs.close();
        }
    }

    public static float[] to1D(float[][] a, int width) {
        float[] r = new float[a.length * width];
        
        for(int i=0; i < a.length; i++)
            System.arraycopy(a[i], 0, r, i * width, width);
        
        return r;
    }
    
    public static float[][] to2D(float[] a, int count, int width) {
        float[][] r = new float[count][width];
        
        for(int i=0; i < count; i++)
            System.arraycopy(a, i * width, r[i], 0, width);
        
        return r;
    }
    
    public static byte[] toByteArray(float[] array) {
        ByteBuffer buffer = ByteBuffer.allocate(array.length * 4);
        for(float f : array)
            buffer.putFloat(f);
        
        return buffer.array();        
    }
    public static float[] toFloatArray(byte[] array) {
        float[] result = new float[array.length / 4];
        
        ByteBuffer.wrap(array).asFloatBuffer().get(result);

        return result;
    }
    
    public static byte[] compress(byte[] bytes) {
        byte[] buffer = new byte[1024];
        ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
        Deflater compressor = new Deflater(Deflater.BEST_COMPRESSION);
        compressor.setInput(bytes);
        compressor.finish();

        while(!compressor.finished()) {
            int compressedLength = compressor.deflate(buffer);
            byteStream.write(buffer, 0, compressedLength);
        }

        return byteStream.toByteArray();
    }
    
    /* compression */
    public static byte[] decompress(byte[] bytes) {
        ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
        
        Inflater decompresser = new Inflater();
        decompresser.setInput(bytes);
        byte[] uncompressed = new byte[0x100000];
        
        while(!decompresser.finished()) {
            try {
                int size = decompresser.inflate(uncompressed);
                byteStream.write(uncompressed, 0, size);
            } catch (DataFormatException ex) {
                decompresser.end();
                throw new RuntimeException(ex);
            }
        }
        
        decompresser.end();
        return byteStream.toByteArray();
    }
    

    /* base 64 stuff */
    final static String base64chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    final static String base64safe  = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";    
    protected static final byte[] base64charsLookup = new byte[0x100];
    static {
        for(int i=0; i < base64charsLookup.length; i++)
            base64charsLookup[i] = -1;

        for(byte i=0; i < base64chars.length(); i++)
            base64charsLookup[base64chars.charAt(i)] = i;

        /* using url-safe base64 '+' -> '-'  and  '/' -> '_' */
        base64charsLookup['-'] = base64charsLookup['+'];
        base64charsLookup['_'] = base64charsLookup['/'];
    }
    public static byte[] base64decode(String base64) {
        /* check for null */
        if(base64 == null || base64.length() == 0)
           return null;
        
        /* check length requirements */
        if(base64.length() % 4 == 1)
           return null;

        int pad = base64.charAt(base64.length()-2) == '=' ? 2 :
                  base64.charAt(base64.length()-1) == '=' ? 1 : 0;
        
        char[] base64charArray = base64.toCharArray();
        byte[] decodedBinary = new byte[(base64charArray.length-pad)*3/4];
        
        int outputIndex = 0;
        int len = base64charArray.length-pad;
        /* each 4 base64 digits represent 3 chars */
        for(int i=0; i < len-3; i+= 4) {
            short a = base64charsLookup[base64charArray[i]];
            short b = base64charsLookup[base64charArray[i+1]];
            short c = base64charsLookup[base64charArray[i+2]];
            short d = base64charsLookup[base64charArray[i+3]];

            if(a < 0 || b < 0 || c < 0 || d < 0)
               return null;

            decodedBinary[outputIndex++] = (byte)(a<<2 | b>>4);
            decodedBinary[outputIndex++] = (byte)(b<<4 | c>>2);
            decodedBinary[outputIndex++] = (byte)(c<<6 | d);
        }

        /* last 3-2 digits */
        int rem = len % 4;
        if(rem != 0) {
           short a = base64charsLookup[base64charArray[len-rem]];
           short b = base64charsLookup[base64charArray[len-rem+1]];
           if(a < 0 || b < 0)
              return null;
           decodedBinary[outputIndex++] = (byte)(a<<2 | b>>4);

           if(rem == 3) {
              short c = base64charsLookup[base64charArray[len-1]];
              if(c < 0)
                 return null;
              decodedBinary[outputIndex++] = (byte)(b<<4 | c>>2);
           }
        }
        return decodedBinary;
    }    
    public static String base64encode(byte[] binary) {
        return base64encode(binary, true);
    }
    public static String base64encode(byte[] binary, boolean safe) {
        StringBuilder builder = new StringBuilder();
        int len = binary.length;
        String base64 = safe ? base64safe:base64chars;

        /* each 3 chars represents 4 base64 digits */
        /* 000000-00 0000-0000 00-000000 */
        for(int i=0; i < len-2; i+= 3) {
            int a = (binary[i]   & 0xff) >>> 2;
            int b = (binary[i]   & 0x3) << 4 | ((binary[i+1] & 0xff) >>> 4);
            int c = (binary[i+1] & 0xf) << 2 | ((binary[i+2] & 0xff) >>> 6);
            int d =  binary[i+2] & 0x3F;

            builder.append(base64.charAt(a));
            builder.append(base64.charAt(b));
            builder.append(base64.charAt(c));
            builder.append(base64.charAt(d));
        }

        /* check that we have all digits */
        int rem = len % 3;
        if(rem != 0) {
           int a = (binary[len-rem] & 0xff) >>> 2;
           builder.append(base64.charAt(a));
           if(rem == 2) {
              int b = (binary[len-2] & 0x3) << 4 | ((binary[len-1] &0xff) >>> 4);
              int c = (binary[len-1] & 0xf) << 2;
              builder.append(base64.charAt(b));
              builder.append(base64.charAt(c));
              if(!safe)
                 builder.append("=");
           } else {
              int b = (binary[len-1] & 0x3) << 4;
              builder.append(base64.charAt(b));
              if(!safe)
                 builder.append("==");
           }
        }

        return builder.toString();
    }    
}
