// loading csv
var lineStart = -1;
var lineEnd = -1;
var currentSampleIndex = -1;
var currentSamples = [];

// canvas drawing
var canvas = document.getElementById("img-canvas");
var ctx = canvas.getContext("2d");
var mode = undefined;
var blind_mode = false;
var outlineStrokeSize = 5;
var boundingBox = [];
var frameOfRefs = {};
// For annotate mode
var currentRelationIndex = {};
var doneSamples = new Set();
// counting samples
var acceptedSamples = new Set();
var rejectedSamples = new Set();

$(document).ready(function() {

    canvas.addEventListener("mousedown", draw, false);
    
    $("#csv-file-select").submit(function(event) {
        event.preventDefault();
        
        var input = $("#csv-file-selector")[0];

        if (Object.keys(frameOfRefs).length < currentSamples.length) {
            var r = confirm("You have not exhausted loaded samples."
                          + " Are you sure you want to load more? This"
                          + " will erase previously loaded ones.");
            if (r != true) {
                return;
            }
        }
        
        if (input.files && input.files[0]) {
            file = input.files[0];
            console.log(file)
            if (!file.name.endsWith(".csv")) {
                alert("File needs to be a CSV.");
                return;
            }
            var tmp_lineStart = parseInt($("#line_start").val());
            var tmp_lineEnd = parseInt($("#line_end").val());
            var arg_keyword = ""
            if ($("#keyword").val() && $("#keyword").val().trim().length > 0) {
                arg_keyword = "&keyword=" + $("#keyword").val().trim()
            }
            var formData = new FormData();
            formData.append('CsvFile', input.files[0]);               
            $.ajax({
                url: "http://localhost:8000/upload-csv?line_start=" + tmp_lineStart + "&line_end=" + tmp_lineEnd + arg_keyword,
                type: "POST",
                data: formData,
                processData: false,
                contentType: false,
                beforeSend: function() {
                    $("#alert").html("<b>Loading data. Waiting for Response...</b>")
                },
                success: function(result) {
                    currentSamples = JSON.parse(result);
                    
                    // Only update the lineStart/lineEnd variables when loading succeeds.
                    lineStart = tmp_lineStart;
                    lineEnd = tmp_lineEnd;
                    var linesLoaded = lineEnd - lineStart;
                    $("#line_start").val(lineEnd);
                    $("#line_end").val(lineEnd + linesLoaded);
                    
                    // Reset global variables
                    currentSampleIndex = 0;
                    frameOfRefs = {};
                    currentRelationIndex = {};
                    doneSamples = new Set();

                    loadSample(currentSampleIndex);
                    console.log(result);
                    console.log();
                }
            })
        } else {
            alert("Choose a file!");
        }
    });
});

function loadSample(index) {
    var sample = currentSamples[index];
    var map_name = sample["map_name"];
    console.log(map_name);
    if (blind_mode) {
        // will not show the image in the map link, but show
        // the image of just the map.
        $("#map-img-display").attr("src", "plotted_map_images/" + map_name + ".PNG");
    } else {
        $("#map-img-display").attr("src", sample["map_link"]);
    }
    $("#hint").html(sample["hint"]);
    var sg = sample["sg"];
    $("#hint-parsed").html(sg["lang"]);
    $("#sg").html("<pre>" + JSON.stringify(sg, null, 4) + "</pre>")
    $("#cur_sample_id").html(index+1);
    $("#total_samples_batch").html(currentSamples.length);
    
    // Mark accept/reject button
    if (acceptedSamples.has(lineStart + index)) {
        $("#accept-btn").addClass("btn-selected");
        $("#reject-btn").removeClass("btn-selected");
    } else {
        $("#reject-btn").addClass("btn-selected");
        $("#accept-btn").removeClass("btn-selected");
        rejectedSamples.add(lineStart + index);
    }
    $("#rejected_num").html(rejectedSamples.size);
    $("#accepted_num").html(acceptedSamples.size);

    // Make the canvas equally large
    var img = document.getElementById("map-img-display");
    imgWidth = img.clientWidth;
    imgHeight = img.clientHeight;
    console.log(img);
    if (imgWidth == 0) {
        $("#alert").html("<b>Alert! Image width not read properly. Press \"Next\" then \"Prev\" to fix it.</b>")
    } else {
        $("#alert").html("")
    }
    $("#img-canvas").attr('width', imgWidth);
    $("#img-canvas").attr('height', imgHeight);
}

function NextSample() {
    if (currentSampleIndex + 1 < currentSamples.length) {
        currentSampleIndex += 1;
        loadSample(currentSampleIndex);
        checkAnnotationStatus();
        redraw();
    } else {
        alert("No more samples!");
    }
}

function PrevSample() {
    if (currentSampleIndex - 1 >= 0) {
        currentSampleIndex -= 1;
        loadSample(currentSampleIndex);
        checkAnnotationStatus();
        redraw();
    } else {
        alert("Cannot go back!");
    }
}

function checkAnnotationStatus() {
    if (doneSamples.has(currentSampleIndex)) {
        $("#alert").html("<b>Annotation for this sample is Done.</b>");
        return;
    } else {
        var cri = currentRelationIndex[currentSampleIndex];
        if (cri === undefined) {
            $("#info").html("");
            return;
        }
        console.log(currentSampleIndex);
        var sample = currentSamples[currentSampleIndex];
        var sg = sample["sg"];        
        var sg_rel = sg["relations"][cri];
        if (sg_rel !== undefined) {
            $("#info").html("<b>" + sg_rel[0] + " " + sg_rel[2] + " " + sg_rel[1] + "</b>");
        }
    }
}

function AcceptSample() {
    if (currentSampleIndex >= 0) {
        acceptedSamples.add(lineStart + currentSampleIndex);
        rejectedSamples.delete(lineStart + currentSampleIndex);        
        $("#rejected_num").html(rejectedSamples.size);
        $("#accepted_num").html(acceptedSamples.size);
        $("#accept-btn").addClass("btn-selected");
        $("#reject-btn").removeClass("btn-selected");        
    }
}

function RejectSample() {
    acceptedSamples.delete(lineStart + currentSampleIndex);    
    rejectedSamples.add(lineStart + currentSampleIndex);
    $("#rejected_num").html(rejectedSamples.size);
    $("#accepted_num").html(acceptedSamples.size);
    $("#reject-btn").addClass("btn-selected");
    $("#accept-btn").removeClass("btn-selected");            
}

function SaveAll() {
    // Create an object containing the processed spatial graphs, then download it as a JSON file.
    var output = {"metadata":{}, "samples":[]};
    output["metadata"]["sample_rows"] = [];
    if (boundingBox.length != 2) {
        alert("You must draw a bounding box!");
        return;
    }
    output["metadata"]["pixel_origin"] = boundingBox[0];
    output["metadata"]["img_width"] = boundingBox[1][0] - boundingBox[0][0];
    output["metadata"]["img_length"] = boundingBox[1][1] - boundingBox[0][1];
    doneSamples.forEach(function(sampleIndex) {
        if (acceptedSamples.has(lineStart + sampleIndex)){
            var sample = currentSamples[sampleIndex];
            var sampleClone = JSON.parse(JSON.stringify(sample));
            sampleClone["sg"]["frame_of_refs_pixels"] = frameOfRefs[sampleIndex];
            output["samples"].push(sampleClone);
            output["metadata"]["sample_rows"].push(lineStart + sampleIndex);
        }
    });

    // Send the output object to server for processing (pixel -> grid cell)
    $.ajax({
        url: "http://localhost:8000/save",
        type: "POST",
        data: JSON.stringify(output),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        async: false,  // block here.
        beforeSend: function() {
            $("#alert").html("<b>Uploading annotated data to Server...</b>")
        },
        success: function(result) {
            console.log(200);
            console.log(result);
            $("#alert").html("<b>Received Served Response. Downloading data file...</b>")
            downloadFileForObject(result);
        },
        fail: function(result) {
            console.log(500);
            $("#alert").html("<b>500 Internal Server Error. File still saving.</b>");
            downloadFileForObject(result);            
        }        
    });
}

function downloadFileForObject(obj) {
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(obj));
    // download json file
    var downloadAnchor = document.getElementById('download-anchor-elem');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "annotated_samples.json");
    downloadAnchor.click();    
}

function ToggleSetBox() {
    if (mode !== undefined && mode !== "set_box") {
        alert("Mode is at " + mode + "; Cannot switch to set box.");
        return;
    }    
    if (mode !== "set_box") {
        if (currentSampleIndex < 0) {
            return;
        }
        $("#set-box-btn").addClass("btn-selected");
        mode = "set_box";
    } else {
        mode = undefined;
        $("#set-box-btn").removeClass("btn-selected");
    }
}

function ToggleAnnotate() {
    if (mode !== undefined && mode !== "annotate") {
        alert("Mode is at " + mode + "; Cannot switch to annotate.");
        return;
    }
    if (mode !== "annotate") {
        if (currentSampleIndex < 0) {
            return;
        }
        $("#annotate-btn").addClass("btn-selected");
        // Make a note of the relation that is being annotated
        var sample = currentSamples[currentSampleIndex];
        var sg = sample["sg"];
        var cri = currentRelationIndex[currentSampleIndex];
        if (cri === undefined) {
            currentRelationIndex[currentSampleIndex] = 0;
            cri = 0;
        }
        var sg_rel = sg["relations"][cri];
        if (sg_rel !== undefined) {
            $("#info").html("<b>" + sg_rel[0] + " " + sg_rel[2] + " " + sg_rel[1] + "</b>");
        }
        mode = "annotate";
    } else {
        mode = undefined;
        $("#annotate-btn").removeClass("btn-selected");
    }
}

function ToggleBlindMode(event) {
    event.preventDefault();
    if (!blind_mode) {
        blind_mode = true;
        $("#blind-mode-btn").addClass("btn-selected");
    } else {
        if (currentSamples.length > 0) {
            alert("Cannot unset blind mode because you already loaded maps!");
            return
        }
        blind_mode = false;
        $("#blind-mode-btn").removeClass("btn-selected");
    }
}



function ClearSampleAnnotation() {
    if (currentSampleIndex in frameOfRefs) {
        frameOfRefs[currentSampleIndex] = [];
        currentRelationIndex[currentSampleIndex] = 0;
        doneSamples.delete(currentSampleIndex);
        $("#alert").html("<b>Cleared Sample Annotations.</b>");
        checkAnnotationStatus()
        redraw();
    }
}


// CANVAS
function draw(event) {
    // Fired when mouse down
    var canvas_x = event.pageX;
    var canvas_y = event.pageY;

    if (mode === "set_box") {
        addBoundingBoxPoint(canvas_x, canvas_y);
        drawBoundingBox();

    } else if (mode === "annotate") {
        if (doneSamples.has(currentSampleIndex)) {
            $("#alert").html("<b>Annotation for this sample is Done.</b>");
            return;
        }
        
        if (!(currentSampleIndex in frameOfRefs)) {
            frameOfRefs[currentSampleIndex] = [];
            currentRelationIndex[currentSampleIndex] = 0;
        }
        var cri = currentRelationIndex[currentSampleIndex];
        var sample = currentSamples[currentSampleIndex];
        var sg = sample["sg"];
        if (frameOfRefs[currentSampleIndex].length <= cri) {
            frameOfRefs[currentSampleIndex].push([]);
        }
        if (cri >= sg["relations"].length) {
            // We are done with this example
            $("#alert").html("<b>Annotation for this sample is Done.</b>");
            $("#info").html("");
            doneSamples.add(currentSampleIndex);
            return;
        }
        var sg_rel = sg["relations"][cri];
        $("#info").html("<b>" + sg_rel[0] + " " + sg_rel[2] + " " + sg_rel[1] + "</b>");
        
        var done = makeNewFrameOfRef(frameOfRefs[currentSampleIndex][cri],
                                     canvas_x, canvas_y);
        if (done) {
            currentRelationIndex[currentSampleIndex] += 1;
            $("#alert").html("<b>One relation annotated.</b>");
            // Make a note of the relation that is being annotated NEXT
            var next_cri = currentRelationIndex[currentSampleIndex];
            if (next_cri >= sg["relations"].length) {
                // We are done with this example
                $("#alert").html("<b>Annotation for this sample is Done.</b>");
                $("#info").html("");
                doneSamples.add(currentSampleIndex);
                return;
            }   
            var sg_rel = sg["relations"][next_cri];
            $("#info").html("<b>" + sg_rel[0] + " " + sg_rel[2] + " " + sg_rel[1] + "</b>");
        }
    }
}

function addBoundingBoxPoint(canvas_x, canvas_y) {
    if (boundingBox.length == 0) {
        boundingBox.push([canvas_x, canvas_y]);
    } else if (boundingBox.length == 1) {
        boundingBox.push([canvas_x, canvas_y]);
    } else {
        boundingBox = [];
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        boundingBox.push([canvas_x, canvas_y]);
    }
}

function drawBoundingBox() {
    var rectSize = outlineStrokeSize;
    ctx.fillStyle="#0066ff";
    ctx.strokeStyle="#0066ff";
    ctx.lineWidth = rectSize;

    if (boundingBox.length == 0) {
        return;
    } else if (boundingBox.length == 1) {
        var canvas_x = boundingBox[0][0]
        var canvas_y = boundingBox[0][1]
        ctx.fillRect(canvas_x - rectSize/2, canvas_y - rectSize/2, rectSize, rectSize);
    } else if (boundingBox.length == 2) {
        // Draw a rectangle
        var first_pt = boundingBox[0];
        var second_pt = boundingBox[1];
        ctx.strokeRect(first_pt[0], first_pt[1],
                       second_pt[0] - first_pt[0], second_pt[1] - first_pt[1])
    } else {
        alert("Unexpected " + boundingBox);
    }
    console.log(boundingBox);
}

function frameOfRefToPts(foref) {
    var first_pt = foref[0];
    var angle = foref[1];
    var diffx = 150;
    var rot_pt = rotateAround([first_pt[0] + diffx, first_pt[1]], first_pt, angle);
    return [first_pt, rot_pt];
}

function ptsToFrameOfRef(pt1, pt2) {
    // Use the +x direction as 0 degree, calculate the degree between front and +x direction.
    var diffx = pt2[0] - pt1[0];
    var diffy = pt2[1] - pt1[1];
    var angle = Math.atan2(diffy, diffx);
    return [pt1, angle];
}

// Draw the frame of references for current sample
function drawFrameOfRefs() {
    var rectSize = outlineStrokeSize;
    var rectSize2 = rectSize * 3;
    ctx.fillStyle="#ff33ff";
    ctx.strokeStyle="#ff33ff";
    ctx.lineWidth = rectSize;

    if (frameOfRefs[currentSampleIndex] === undefined) {
        return;
    }
    frameOfRefs[currentSampleIndex].forEach(function(foref, index){
        if (foref.length == 1) {
            var first_pt = foref[0];
            ctx.fillRect(first_pt[0] - rectSize/2, first_pt[1] - rectSize/2, rectSize, rectSize);
        } else if (foref.length == 2) {
            // Draw first point then draw the vectors
            var pts = frameOfRefToPts(foref);
            var first_pt = pts[0];
            var second_pt = pts[1];
            ctx.fillRect(first_pt[0] - rectSize/2, first_pt[1] - rectSize/2, rectSize, rectSize);            
            drawLine(first_pt[0], first_pt[1], second_pt[0], second_pt[1], outlineStrokeSize, "#ff33ff", ctx);
            ctx.fillStyle="#ff33ff";
            ctx.fillRect(second_pt[0] - rectSize2/2, second_pt[1] - rectSize2/2, rectSize2, rectSize2);

            // Now, draw a line orthogonal to the front direction (indicating "right").
            var rot_pt = rotateAround(second_pt, first_pt, 0.5*Math.PI);
            drawLine(first_pt[0], first_pt[1], rot_pt[0], rot_pt[1], outlineStrokeSize, "#33ff33", ctx);
            ctx.fillStyle="#33ff33";
            ctx.fillRect(rot_pt[0] - rectSize2/2, rot_pt[1] - rectSize2/2, rectSize2, rectSize2);
        }
    });
}

// Input: Frame of reference representation (list), and a point (x,y)
// Output: Continue construction of the frame of reference representation.
function makeNewFrameOfRef(foref, canvas_x, canvas_y) {
    var rectSize = outlineStrokeSize;
    ctx.fillStyle="#ff33ff";
    ctx.strokeStyle="#ff33ff";
    ctx.lineWidth = rectSize;
    
    if (foref.length == 0) {
        // Then canvas_x, canvas_y is the point
        foref.push([canvas_x, canvas_y]);
        ctx.fillRect(canvas_x - rectSize/2, canvas_y - rectSize/2, rectSize, rectSize);
        return false;
        
    } else if (foref.length == 1) {
        // Here, canvas_x, canvas_y suggests the direction (angle) of the FoR
        var front = [foref[0], [canvas_x, canvas_y]]
        drawLine(front[0][0], front[0][1], front[1][0], front[1][1], outlineStrokeSize, "#ff33ff", ctx);
        rectSize2 = rectSize * 3;
        ctx.fillRect(canvas_x - rectSize2/2, canvas_y - rectSize2/2, rectSize2, rectSize2);

        // Now, draw a line orthogonal to the front direction (indicating "right").
        var rot_pt = rotateAround([canvas_x, canvas_y], foref[0], 0.5*Math.PI);
        drawLine(front[0][0], front[0][1], rot_pt[0], rot_pt[1], outlineStrokeSize, "#33ff33", ctx);
        ctx.fillStyle="#33ff33";
        ctx.fillRect(rot_pt[0] - rectSize2/2, rot_pt[1] - rectSize2/2, rectSize2, rectSize2);

        // Use the +x direction as 0 degree, calculate the degree between front and +x direction.
        var angle = ptsToFrameOfRef(foref[0], [canvas_x, canvas_y])[1];
        var pts = frameOfRefToPts([foref[0], angle]);
        console.log("Drawn Points: ");
        console.log("    " + foref[0]);
        console.log("    " + [canvas_x, canvas_y]);
        console.log(" angle: " + angle);        
        console.log("Computed points: ");
        console.log("    " + pts[0]);
        console.log("    " + pts[1]);
        console.log(" angle: " + ptsToFrameOfRef(pts[0], pts[1])[1]);
        console.log("---");
        foref.push(angle);
        return true;
    } else {
        return true;
    }
}



function redraw() {
    // Clear everything. Then redraw the bounding box, and the Frame of Refs for the current sample.
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawBoundingBox();
    drawFrameOfRefs();
}


function drawLine(x1, y1, x2, y2, width, style, ctx) {
    ctx.lineWidth = width;
    ctx.strokeStyle = style;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}


// Computes Y=AB. Requires that matrix A is mxn, matrix B is nxr. Returns Y.
function matXmat(A, B, m, n, r) {
    var Y = [];
    for (var i = 0; i < m; i++) {
	for (var j = 0; j < r; j++) {
	    var sum = 0.0;
	    for (var k = 0; k < n; k++) {
		sum += A[i*m+k] * B[k*r+j];
	    }
	    Y.push(sum);
	}
    }
    return Y;
}

// Return a matrix that does affine translation by given vector v.
function affine_translate(v) {
    return [
	1, 0, v[0],
	0, 1, v[1],
	0, 0, 1
    ];
}

// Return a matrix that does affine rotation around the origin by given angle theta (radian).
function affine_rotate(theta) {
    return [
	Math.cos(theta), -Math.sin(theta), 0,
	Math.sin(theta),  Math.cos(theta), 0,
	0,                0,               1
    ];
}


// Rotate point p around point q by angle theta (radian).
function rotateAround(p, q, theta) {
    // We are doing the following:
    // 1. Translate q to origin   T(q)
    // 2. Rotate                  R(theta)
    // 3. Translate back          T(-q)
    // Combined: T(-q)R(theta)T(q)

    var Tq = affine_translate(q);
    var Rt = affine_rotate(theta);
    var Tnq = affine_translate([-q[0], -q[1]]);

    var M = matXmat(matXmat(Tq,Rt,3,3,3), Tnq,3,3,3);

    var pa = [p[0], p[1], 1];
    var p_rot = matXmat(M, pa, 3, 3, 1);
    return p_rot;
}
