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
package org.fjnn.cuda;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.net.URLConnection;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class CudaModule {
    public static final String MODULE_MATRIX = "matrix";
    public static final String MODULE_ACTIVATION = "activation";    
    public static final String MODULE_GENETIC = "genetic";    
    public static final String MODULE_ACCUMULATE = "accumulate";    
    public static final String UTIL_FILE = "util.h";
    
    private static final String CUDA_COMPILER = "nvcc";
    
    private static final Map<String, String> ptxFiles = new HashMap<>();
    
    private static final String CUDA_DIRECTORY = "cuda";
    
    private static final String PTX_FILE_EXTENTION = ".ptx";
    
    CUmodule module;
    HashMap<String, CUfunction> functions;

    protected CudaModule(String name) throws IOException {
        String path = getPtxFile(name);
        
        this.module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, path);

        functions = new HashMap<>();
    }

    protected CUfunction getFunction(String functionName) {
        if(!functions.containsKey(functionName))
            loadFunction(functionName);
        
        return functions.get(functionName);
    }

    private synchronized void loadFunction(String functionName) {
        if(functions.containsKey(functionName))
            return;

        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, functionName);

        functions.put(functionName, function);
    }
    
    /********************/
    /* Static functions */
    /********************/
    private static synchronized String getPtxFile(String name) throws IOException {
        if(ptxFiles.containsKey(name))
            return ptxFiles.get(name);
        
        /* Make sure to get folder for the current jar */
        String cDir = new File("").getAbsolutePath();
        
        String cudaDir = cDir + "/" + CUDA_DIRECTORY;

        File folder = new File(cudaDir);
        
        if(!folder.exists() || !folder.isDirectory())
            folder.mkdir();

        ensureSourceExists(cudaDir, name);
        
        String cuPath = cudaDir + "/" + name + ".cu";
        String ptxPath = cudaDir + "/" + name + PTX_FILE_EXTENTION;
        File ptx = new File(ptxPath);
        
        if(ptx.exists() && ptx.lastModified() > new File(cuPath).lastModified()) {
            ptxFiles.put(name, ptxPath);
            return ptxPath;
        }

        compileCU(cuPath, ptxPath);

        ptxFiles.put(name, ptxPath);
        
        return ptxPath;
    }
    
    /**
     * Compiles CU files, throws runtime exception if anything went wrong
     */ 
    static void compileCU(String cuPath, String ptxPath) {
        File cuFile = new File(cuPath);
        
        if(!cuFile.exists())
            throw new RuntimeException("Cuda file not found: " + cuPath);
        
        String arch = System.getProperty("sun.arch.data.model");
        
        String command = String.format("%s -ptx %s -o %s -m %s -lineinfo",
                                        CUDA_COMPILER, cuPath, ptxPath, arch);

        System.out.println("Compiling: \n" + command);

        try {
            Process process = Runtime.getRuntime().exec(command);
            
            int result = process.waitFor();
            
            if(result != 0) {
                String output = util.readStream(process.getInputStream());
                String error  = util.readStream(process.getErrorStream());
                
                System.out.println("nvcc exit: " + result);
                System.out.println("outputMessage:\n" + output);
                System.out.println("errorMessage:\n" + error);
                
                throw new RuntimeException("Unable to create .ptx file: "+ error);
            }
        } catch(IOException | InterruptedException ex) {
            throw new RuntimeException(ex);
        }
    }
    
    private static void ensureSourceExists(String path, String name) throws IOException {
        String cuPath = path + "/" + name + ".cu";
        
        URL url = CudaModule.class.getResource("/" + CUDA_DIRECTORY + "/" + name + ".cu");
        
        URLConnection connection = url.openConnection();
                
        File current = new File(cuPath);

        if(current.exists() && current.lastModified() >= connection.getLastModified())
           return;
        
        String file = util.readStream(connection.getInputStream());
        
        util.saveFile(cuPath, file.getBytes("UTF-8"));
    }
    
    public static void saveUtilFile(int threadsPerBlock) throws IOException {
        /* Make sure to get folder for the current jar */
        String cDir = new File("").getAbsolutePath();
        
        String cudaDir = cDir + "/" + CUDA_DIRECTORY;
        
        String filename = cudaDir + "/" + UTIL_FILE;
        File current = new File(filename);
        
        URL url = CudaModule.class.getResource("/" + CUDA_DIRECTORY + "/" + UTIL_FILE);
        
        URLConnection connection = url.openConnection();
        
        if(current.exists() && current.lastModified() >= connection.getLastModified()) {
            String file = new String(util.readFile(filename));
            Pattern p = Pattern.compile("#define THREADS_PER_BLOCK ([0-9]+)");
            Matcher m = p.matcher(file);

            if(m.find()) {
                int currentThreadsPerBlock = Integer.parseInt(m.group(1));
                
                if(threadsPerBlock == currentThreadsPerBlock)
                    return;
            }
        }
        
        /* delete all ptx files */
        File dir = new File(cudaDir);
        
        if(!dir.exists() || !dir.isDirectory())
            dir.mkdir();
        
        for(File f : dir.listFiles()) {
            if(f.getName().endsWith(PTX_FILE_EXTENTION)) {
                System.out.println("Deleting: " + f.getName());
                f.delete();
            }
        }
        
        String file = util.readStream(connection.getInputStream());
        
        file = file.replace("INSERT_THREADS_PER_BLOCK", Integer.toString(threadsPerBlock));
        
        util.saveFile(filename, file.getBytes("UTF-8"));
    }
}
