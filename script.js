function loadPdf() {
    const pdfInput = document.getElementById("pdfInput");
    const pdfOutput = document.getElementById("pdfOutput");
    const file = pdfInput.files[0];

    if (file) {
        const fileReader = new FileReader();

        fileReader.onload = function() {
            const typedarray = new Uint8Array(this.result);
            pdfjsLib.getDocument(typedarray).promise.then(function(pdf) {
                pdf.getPage(1).then(function(page) {
                    page.getTextContent().then(function(content) {
                        let textContent = '';
                        content.items.forEach(function (item) {
                            textContent += item.str + ' ';
                        });
                        pdfOutput.value = textContent;
                    });
                });
            });
        };

        fileReader.readAsArrayBuffer(file);
    }
}