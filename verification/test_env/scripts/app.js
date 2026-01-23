
export const app = {
    extensions: [],
    registerExtension: function(extension) {
        console.log("Extension registered:", extension.name);
        this.extensions.push(extension);
        if (extension.setup) {
            extension.setup(this);
        }
    }
};

// Mock the graph canvas global
window.LGraphCanvas = class {
    constructor() {}
};
window.LGraphCanvas.prototype.drawNodeWidgets = function() {};
