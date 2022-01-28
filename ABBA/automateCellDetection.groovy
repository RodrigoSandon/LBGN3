import static qupath.lib.gui.scripting.QPEx.*
import ch.epfl.biop.qupath.atlas.allen.api.AtlasTools
import qupath.lib.images.ImageData
import groovy.io.FileType
import java.awt.image.BufferedImage
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.gui.commands.ProjectCommands
import qupath.lib.regions.ImagePlane
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathCellObject

clearAllObjects()

def project = getProject()
def count = 0
for (entry in project.getImageList()){
    
    //ImageData imageData = getCurrentImageData();
    ImageData imageData = entry.readImageData() // figure out how to work on every image separately (or try the batch statement)
    def hierarchy = imageData.getHierarchy()
    print imageData
    //print imageData.getClass()
    //print hierarchy
    
    //Import transformed atlas
    ch.epfl.biop.qupath.atlas.allen.api.AtlasTools.loadWarpedAtlasAnnotations(imageData, true);
    
    ROIS = []
    def annotations = hierarchy.getAnnotationObjects()
    //print "Annotations: " + annotations
    // Test on just one roi
    //Find specific object region (cerebral cortex)
    def roi = annotations[50]
    
    for (i in annotations) {
        def roiAsString = i.toString()
        if (roiAsString.contains("Orbital area, medial part") == true) {
            roi = i
        }
    }
    
    print "ROI: " + roi.toString()
//    print roi.getROI()
//    //print("ROI DESCRIPTION (str):" + roi.getDescription())
    ROIS.add(roi)
    
    // For all ROIS
//    for (i in annotations) {
//        ROIS.add(i)
//    }
    
    // Loop annotations in a given hierarchy
    
    print "List of ROIS: " + ROIS
    print "Amount of ROIS to process: " + ROIS.size().toString()
    
    for (ann in ROIS){ //if want all regions, just iterate through the "annotations" variable
        // Select annotation  
        hierarchy.getSelectionModel().clearSelection()
        hierarchy.getSelectionModel().setSelectedObject(ann)
        
        // Run Cell Detection in given annotation
        runPlugin('qupath.imagej.detect.cells.PositiveCellDetection', imageData, '{"detectionImage": "FITC",  "requestedPixelSizeMicrons": 0,  "backgroundRadiusMicrons": 10.0,  "medianRadiusMicrons": 0.0,  "sigmaMicrons": 2.0,  "minAreaMicrons": 55.0,  "maxAreaMicrons": 400.0,  "threshold": 15.0,  "watershedPostProcess": true,  "cellExpansionMicrons": 0.0,  "includeNuclei": false,  "smoothBoundaries": true,  "makeMeasurements": true,  "thresholdCompartment": "Cytoplasm: FITC mean",  "thresholdPositive1": 5.0,  "thresholdPositive2": 10.0,  "thresholdPositive3": 15.0,  "singleThreshold": true}');
        
        // save ImageData to image
        entry.saveImageData(imageData)
    }  
    
    // Get name of current image    
    def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
    print "NAME OF CURRENT IMG: " + name
    // Save Detection Measurements 
    path_to_save = "/Volumes/T7Touch/NIHBehavioralNeuroscience/ABBA/p1/results"
    
    def exportType = PathCellObject.class
    def outputPath=path_to_save+'/'+ name+ '_' + roi.toString() +'.csv' 
    saveDetectionMeasurements(imageData,outputPath)
    count += 1
    
    if (count == 1) {
        break
    }
}
