// Sample chart data (you can replace this with dynamic data from Flask)
const sampleLabels = ['January', 'February', 'March', 'April', 'May'];
const sampleData = [65, 59, 80, 81, 56];

// Bar Chart
const barCtx = document.getElementById('barChart').getContext('2d');
new Chart(barCtx, {
    type: 'bar',
    data: {
        labels: sampleLabels,
        datasets: [{
            label: 'Sales',
            data: sampleData,
            backgroundColor: '#007bff'
        }]
    }
});

// Line Chart
const lineCtx = document.getElementById('lineChart').getContext('2d');
new Chart(lineCtx, {
    type: 'line',
    data: {
        labels: sampleLabels,
        datasets: [{
            label: 'Revenue',
            data: sampleData,
            borderColor: '#28a745',
            fill: false
        }]
    }
});

// Pie Chart
const pieCtx = document.getElementById('pieChart').getContext('2d');
new Chart(pieCtx, {
    type: 'pie',
    data: {
        labels: ['Red', 'Blue', 'Yellow', 'Green'],
        datasets: [{
            data: [300, 50, 100, 150],
            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#28a745']
        }]
    }
});

// Scatter Plot
const scatterCtx = document.getElementById('scatterChart').getContext('2d');
new Chart(scatterCtx, {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Random Points',
            data: [{x: 10, y: 20}, {x: 15, y: 10}, {x: 20, y: 30}],
            backgroundColor: '#ff6384'
        }]
    }
});