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
import java.util.HashMap;
import java.util.Map;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

/**
 *
 * @author ahmed
 */
public class CudaModule {
    public static final String MODULE_MATRIX = "matrix";
    public static final String MODULE_ACTIVATION = "activation";    
    
    
    CUmodule module;
    HashMap<String, CUfunction> functions;

    protected CudaModule(String name) {
        String path = getCudaPtxFile(name);
        
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
    private static Map<String, String> ptxFiles = new HashMap<>();
    
    private static final String CUDA_PTX_DIRECTORY = "ptx";
    private static final String CUDA_SRC_DIRECTORY = "cuda";
    
    private static synchronized String getCudaPtxFile(String name) {
        if(ptxFiles.containsKey(name))
            return ptxFiles.get(name);
        
        /* Make sure to get folder for the current jar */
        String path = new File("").getAbsolutePath();//CudaModule.class.getProtectionDomain().getCodeSource().getLocation().getPath();
        
        String ptxDir = path + "/" + CUDA_PTX_DIRECTORY;

        File folder = new File(ptxDir);
        
        if(!folder.exists() || !folder.isDirectory())
            folder.mkdir();

        String ptxPath = ptxDir + "/" + name + ".ptx";
        File ptx = new File(ptxPath);
        
        /**
         * TODO: move cuda sources to jar 
         * CudaModule.class.getResource(CUDA_SRC_DIRECTORY + "/" + name + ".cu").getFile()
         */
        String cuPath = path + "/" + CUDA_SRC_DIRECTORY + "/" + name + ".cu";
        
//        System.out.println(CudaModule.class.getResource("/" + CUDA_SRC_DIRECTORY + "/" + name + ".cu"));

        /* TODO: some checks on the current file */
        if(ptx.exists() && ptx.lastModified() > new File(cuPath).lastModified()) {
            ptxFiles.put(name, ptxPath);
            return ptxPath;
        }

        CudaUtil.compileCU(cuPath, ptxPath);

        ptxFiles.put(name, ptxPath);
        
        return ptxPath;
    }
}
