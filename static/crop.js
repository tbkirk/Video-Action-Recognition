const form = document.getElementById("video_upload");
const frame_canvas = document.getElementById("frame_canvas")
const frame_context = frame_canvas.getContext('2d');
const mask_canvas = document.getElementById("mask_canvas")
const mask_context = mask_canvas.getContext('2d');
const object_list_element = document.getElementById('object_list')
var object_list = []
console.log(object_list)

async function sendData() {
    // Associate the FormData object with the form element
    const formData = new FormData(form);

    try {
        const response = await fetch("/crop/load_video", {
        method: "POST",
        // Set the FormData instance as the request body
        body: formData,
        });
        const img_blob = await response.blob();
        console.log(img_blob);
        first_frame_url = URL.createObjectURL(img_blob);
        img_bitmap = await createImageBitmap(img_blob);
        frame_canvas.height = img_bitmap.height
        frame_canvas.width = img_bitmap.width
        mask_canvas.height = img_bitmap.height
        mask_canvas.width = img_bitmap.width
        frame_context.drawImage(img_bitmap, 0,0)
    } catch (e) {
        console.error(e);
    }
}

// Take over form submission
form.addEventListener("submit", (event) => {
event.preventDefault();
sendData();
});

async function clickCanvas(event){
    // get click location on canvas
    const rect = frame_canvas.getBoundingClientRect();
    const elementRelativeX = event.clientX - rect.left;
    const elementRelativeY = event.clientY - rect.top;
    const canvasRelativeX = Math.floor(elementRelativeX * frame_canvas.width / rect.width);
    const canvasRelativeY = Math.floor(elementRelativeY * frame_canvas.height / rect.height);
    // add some canvas drawing code here

    console.log(canvasRelativeX,canvasRelativeY)
    object_list.push([canvasRelativeX, canvasRelativeY])
    updateList()
}

function updateList(){
    object_list_element.innerHTML=''
    let i = 0
    for (const element of object_list) {
        i++;
        let list_item = document.createElement('ul')
        list_item.innerText = i + ': ' + element[0] + ', ' + element[1]
        list_item.index = i-1
        object_list_element.append(list_item)
    }
}

function resetObjects(){
    object_list = []
    updateList()
}

async function processObjects() {
    let response_body = JSON.stringify( {'target_list': object_list});
    console.log(response_body)
    const response = await fetch("/crop/detect_objects", {
        method: "POST",
        headers: {
            "accept": "application/json",
            "content-type": "application/json"
        },
        body: response_body,
        
    })
    console.log(response)
    const img_blob = await response.blob();
    //console.log(img_blob);
    const img_bitmap = await createImageBitmap(img_blob);
    mask_context.drawImage(img_bitmap, 0,0)    
}

async function track() {
    for (let index = 0; index < 1; index++) { // todo remove loop
        const response = await fetch("/crop/track")
        const coords = await response.json() // an array with dimensions objects * time * xy coordinates
        console.log(coords)
        mask_context.clearRect(0, 0, mask_canvas.width, mask_canvas.height);
        for (const object_coords of coords){
            mask_context.beginPath()
            mask_context.strokeStyle = "red";
            mask_context.lineWidth = 2;
            mask_context.moveTo(object_coords[0][0], object_coords[0][1])
            for (const xy of object_coords) {
                mask_context.lineTo(xy[0], xy[1])
            }
            mask_context.stroke()
        }
        const image_response = await fetch('/crop/get_frame')
        const img_blob = await image_response.blob();
        const img_bitmap = await createImageBitmap(img_blob);
        frame_context.drawImage(img_bitmap, 0,0)
    }
}